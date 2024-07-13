import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from math import sin, cos, fmod, pi, inf, isinf
from typing import Any, Callable

from common import *
from networks import Network, Layer, FFNN, RBFNN, check_weight_gradients

TIMESTEP = 0.1
DISPLAY = True

class ADHDP:
    def __init__(self, p_actor: Network, p_critic: Network) -> None:
        self.actor = p_actor
        self.critic = p_critic

        self.J_prev = 0
        self.gamma = 0.95
        self.error = [0, 0]

        self.memory: tuple = ()

    def learn_step_start(self, state: np.ndarray) -> np.ndarray:
        u, grad_actor = self.actor.get_weight_gradient(state)
        u = u.flatten()
        #u += np.random.uniform(-0.3, 0.3)

        # Cache results for learn_step_end
        self.memory = u, grad_actor
        return u
    
    def learn_step_end(self, state: np.ndarray, reward: float, initial_timestep: bool=False):
        u, grad_actor = self.memory

        J_tgt = 0

        critic_inp = np.array([state[1]/np.pi, state[3]/20, u[0]])
        J, E_c = self.critic.learn_ml(0.2, critic_inp, reward)
        #J, E_c = self.critic.learn_gd(0.0001, critic_inp, reward)
        outp_critic, grad_critic = self.critic.get_weight_gradient(critic_inp)
        J = outp_critic[0,0]

        e_a = np.array([[J]])
        E_a: float = .5 * (e_a.T@e_a).flatten()[0]

        self.error = [E_a, E_c]
        
        # update actor weights
        critic_io_derivatives = self.critic.get_io_gradient(critic_inp)
        x_size = self.actor.get_layer_size(0)
        dJdx = critic_io_derivatives[:,:x_size]
        dJdu = critic_io_derivatives[:,x_size:]

        dxdu = np.zeros((x_size, 1))
        dxdu[0,0] = 0
        dxdu[1,0] = 0#daddot_du(critic_inp[0], critic_inp[2]) * TIMESTEP

        #actor_weight_delta = (grad_actor.T @ e_a @ dJdu + e_a @ dJdx @ dxdu).flatten() * -1e-3
        #self.actor.update_weights(actor_weight_delta)

        #print(J, reward, E_c)

        self.J_prev = J


def cart_pole_dynamics(state: np.ndarray, u: float) -> np.ndarray:
    g0 = 9.81
    mp = 1.0
    mc = 10.0
    Lp = 1.0
    force_multiplier = 1000

    x, a, x_dot, a_dot = state
    F = u * force_multiplier

    a_ddot = ((g0 * sin(a) - cos(a) * (F + mp * Lp * a_dot**2 * sin(a)) / (mp + mc)) /
            # ----------------------------------------------------------------------------
                        (Lp * (4/3 - mp/(mp+mc) * cos(a)**2)))

    x_ddot = (F + mp*Lp * (sin(a)*a_dot**2 - a_ddot*cos(a))) / (mp + mc)

    return np.array([x_dot, a_dot, x_ddot, a_ddot])


def daddot_du(a: float, u: float) -> float:
    g0 = 9.81
    mp = 1.0
    mc = 10.0
    Lp = 1.0

    return cos(a) / (Lp * (mp + mc) * 4/3 - Lp * mp * cos(a)**2)


def plot_critic(adhdp: ADHDP):
    xx, yy = np.meshgrid(np.linspace(-pi, pi, 20), np.linspace(-20, 20, 20))
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            inp = np.array([xx[i, j]/pi, yy[i, j]/20])
            u = adhdp.actor.eval(inp).flatten()
            zz[i, j] = adhdp.critic.eval(np.concatenate((inp, u)))[0,0]
    #plt.axis('scaled')
    #plt.contourf(xx, yy, zz, levels=np.linspace(0,1,10), extend='both')
    plt.contourf(xx, yy, zz)
    plt.colorbar()
    plt.show()
    

def plot_actor(actor: Network):
    xx, yy = np.meshgrid(np.linspace(-pi, pi, 20), np.linspace(-20, 20, 20))
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = actor.eval(np.array([xx[i, j], yy[i, j]]))[0,0]
    #plt.axis('scaled')
    plt.contourf(xx, yy, zz)
    plt.colorbar()
    plt.show()


def visualize_state(state: np.ndarray, adhdp: ADHDP, reward: float) -> None:
    x, a, xdot, adot = state
    W, H = rl.get_screen_width(), rl.get_screen_height()

    Lp = 1.0
    screen_x1 = int(W*(x*0.2+0.5))
    offset = screen_x1 - int(W*0.5)
    screen_y1 = int(H * 0.5)
    head_x = x + Lp * sin(a)
    head_y = Lp * cos(a)
    screen_x2 = int(W*(head_x*0.2+.5))
    screen_y2 = int(H * 0.5 - W*0.2*head_y)

    rl.begin_drawing()
    rl.clear_background(BG)
    rl.draw_rectangle(screen_x1-50-offset, screen_y1, 100, 50, FG)
    rl.draw_circle(screen_x2-offset, screen_y2, 30, FG)
    rl.draw_line_ex(rl.Vector2(screen_x1-offset, screen_y1), rl.Vector2(screen_x2-offset, screen_y2), 10, FG)

    rl.draw_text(f"R = {reward}", 10, 10, 12, HIGHLIGHT)
    rl.draw_text(f"Ec = {adhdp.error[1]}", 10, 20, 12, HIGHLIGHT)

    rl.end_drawing()


def cart_pole_test():

    input_dim = 2
    output_dim = 1
    actor = FFNN([
        Layer.linear(input_dim),
        #Layer.tanh(2),
        Layer.linear(output_dim),
    ], (-1, 1), (-1, 1))
    actor.weights = np.array([
        #1,0,0,
        #0,1,0,
        0.2, 0.1, 0.0  # Proportional gain, Derivative gain, bias
    ])
    
    c_w = 10
    #critic = FFNN([
    #    Layer.linear(input_dim + output_dim),
    #    Layer.relu(2),
    #    Layer.linear(1),
    #], (-c_w, c_w), (-10, 10))
    #ref_critic = FFNN(critic.architecture, (critic.w_min, critic.w_max), (critic.b_min, critic.b_max))
    
    critic = RBFNN.grid_spaced(output_dim,
        np.linspace(-1, 1, 5), 
        np.linspace(-1, 1, 5), 
        np.linspace(-1, 1, 3))
    
    #prep_critic(critic)
    #critic.save_to("saved_networks/pendulum/critic5x5x3_prep.dat")
    critic.load_from("saved_networks/pendulum/critic5x5x3_prep.dat")

    ref_critic = RBFNN(input_dim + output_dim, critic.size, output_dim, (critic.w_min, critic.w_max), (critic.b_min, critic.b_max))
    ref_critic.weights = critic.weights.copy()
    ref_critic.set_rbfs(critic.centers, 1/critic.inv_stdevs)
    
    adhdp = ADHDP(actor, critic)

    #check_gradients(actor)
    #check_gradients(critic)
    if DISPLAY:
        #rl.set_target_fps(100)
        rl.init_window(800, 500, "Pendulum")
    
    epochs = 20

    error_evolution = np.zeros(epochs)
    ref_error_evolution = np.zeros(epochs)

    for i in range(epochs):
        state = np.zeros(4)
        #state[1] = (np.random.uniform(-2, 2) % (2*pi)) - pi
        state[1] = np.pi
        state[3] = np.random.normal(0, 2)

        time = 0
        ref_J_prev = 0
        error_evolution_run = []
        ref_error_evolution_run = []

        while True:
            x, a, x_dot, a_dot = state
            dt = TIMESTEP#rl.get_frame_time()

            #F = 200*a + 100*a_dot# - 10*x - 10*x_dot  # override with PD control to only train critic
            #a = np.random.uniform(-pi, pi)
            #a_dot = np.random.uniform(-20, 20)
            u = adhdp.learn_step_start(np.array([a, a_dot]))[0]

            # Runge-Kutta
                
            k1 = cart_pole_dynamics(state, u)
            k2 = cart_pole_dynamics(state + k1*dt/2, u)
            k3 = cart_pole_dynamics(state + k2*dt/2, u)
            k4 = cart_pole_dynamics(state + k3*dt, u)
            state += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            state[1] = fmod(state[1] + pi, 2*pi) - pi
            state[0] = max(-1, min(1, state[0]))
            if np.any(state > 1e6):
                break
            #x, a, x_dot, a_dot = state

            #print(x, x_dot, a, a_dot)

            #adhdp.learn_step_end(a**2 + np.log(x**2+1))
            reward = np.sqrt((a/pi)**2 + (a_dot/20)**2) / 20
            adhdp.learn_step_end(np.array([0, a, 0, a_dot]), reward, time==0)
            if time > 10 or abs(a_dot) > 20 or reward < 1e-6:
                break

            # Visualization
            if DISPLAY:
                visualize_state(state, adhdp, reward)
                if rl.window_should_close():
                    rl.close_window()
                    return

            time += dt
            error_evolution_run.append(adhdp.error[1])
            #ref_J = ref_critic.eval(np.array([a, a_dot, u]))[0]
            #ref_e_c = ref_J_prev - (adhdp.gamma * ref_J + reward)
            #ref_error_evolution_run.append(0.5 * ref_e_c**2)
            #ref_J_prev = ref_J

        error_evolution[i] = np.average(np.array(error_evolution_run))
        print(i, np.average(np.array(error_evolution_run)), np.median(np.array(error_evolution_run)))
        #ref_error_evolution[i] = np.median(np.array(ref_error_evolution_run))


    #plt.semilogy(ref_error_evolution + 1, label="reference")
    #plt.plot(error_evolution + 1, label="trained")
    #plt.ylim((1, 10))
    #plt.legend()
    #plt.show()

    plot_critic(adhdp)

    if DISPLAY:
        rl.close_window()
    
def prep_critic(critic: Network):
    ''' Trains critic on utility function '''
    N = 100
    ee = np.zeros(N)
    for i in range(N):
        print(i)
        x = np.random.uniform(-1, 1, 3)
        ee[i], _ = critic.learn_ml(0.01, x, float(np.linalg.norm(x[:2])))
        #ee[i] = critic.learn_gd(1e-2, x, float(np.linalg.norm(x[:2])))

    return 

    plt.plot(ee)
    plt.ylim(0, 1)
    plt.show()
    
    print(critic.weights)

    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i,j] = critic.eval(np.array([xx[i,j], yy[i,j], 0]))
            #zz[i,j] = np.sqrt(xx[i,j]**2 + yy[i,j]**2)

    plt.contourf(xx, yy, zz)
    plt.colorbar()
    plt.show()
    
if __name__ == "__main__":
    cart_pole_test()
