import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from math import sin, cos, fmod, pi, inf, isinf
from typing import Any, Callable

from common import *
from networks import Network, Layer, FFNN, RBFNN, levenberg_marquard

TIMESTEP = 0.1
DISPLAY = True
GAMMA = 0.95

class ADHDP:
    def __init__(self, p_actor: Network, p_critic: Network) -> None:
        self.actor = p_actor
        self.critic = p_critic

        self.J_prev = 0
        self.reward_prev = 0
        self.gamma = GAMMA
        self.error = [0, 0]

    def step_and_learn(self, state: np.ndarray, f_dynamcis: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                       f_reward: Callable[[np.ndarray], float], dt: float) -> np.ndarray:
        
        u, grad_actor = self.actor.get_weight_gradient(state)
        u = u.flatten()

        #u += np.random.normal(0, 0.1)

        # Runge-Kutta
        x_size = self.actor.get_layer_size(0)
        integration_state = np.concatenate((state, state*0))
        k1 = f_dynamcis(integration_state, u)
        k2 = f_dynamcis(integration_state + k1*dt/2, u)
        k3 = f_dynamcis(integration_state + k2*dt/2, u)
        k4 = f_dynamcis(integration_state + k3*dt, u)
        prev_integration_state = integration_state
        integration_state += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        integration_state[0] = fmod(integration_state[0] + pi, 2*pi) - pi

        # get reward
        reward = f_reward(integration_state[:x_size])

        critic_inp = np.concatenate((integration_state[:x_size], u))
        critic_inp[0] /= pi
        critic_inp[1] /= 20

        J, E_c = self.critic.learn_ml(0.2, critic_inp, self.gamma * self.J_prev + reward)
        #J, E_c = self.critic.learn_gd(0.001, critic_inp, self.gamma * self.J_prev + reward)
        #J, E_c = self.critic.eval(critic_inp), 0

        J_tgt = 1
        e_a = np.array([[J - J_tgt]])
        E_a: float = .5 * (e_a.T@e_a).flatten()[0]

        self.error = [E_a, E_c]
        
        # update actor weights
        critic_io_derivatives = self.critic.get_io_gradient(critic_inp)
        dJdx = critic_io_derivatives[:,:x_size]
        dJdu = critic_io_derivatives[:,x_size:]
        dxdu = prev_integration_state[x_size:]

        true_grad_actor = (dJdu + dJdx @ dxdu).T @ grad_actor

        actor_weight_delta = (true_grad_actor.T @ e_a).flatten() * -1e-4
        #actor_weight_delta = levenberg_marquard(true_grad_actor, e_a, 0.1)
        #self.actor.update_weights(actor_weight_delta)

        self.reward_prev = reward
        self.J_prev = J

        return integration_state[:x_size]


def cart_pole_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    ''' Also integrates dxdu '''
    g0 = 9.81
    mp = 1.0
    mc = 10.0
    Lp = 1.0
    force_multiplier = 1000

    a, a_dot, da_du, dadot_du = x
    F = u[0] * force_multiplier

    a_ddot = ((g0 * sin(a) - cos(a) * (F + mp * Lp * a_dot**2 * sin(a)) / (mp + mc)) /
            # ----------------------------------------------------------------------------
                        (Lp * (4/3 - mp/(mp+mc) * cos(a)**2)))
    daddot_du = cos(a) / (Lp * (mp + mc) * 4/3 - Lp * mp * cos(a)**2)

    return np.array([a_dot, a_ddot, dadot_du, daddot_du])


def cart_pole_reward(x: np.ndarray) -> float:
    a, adot = x
    return 1 - np.sqrt((a/pi)**2 + (adot/20)**2) / (1 - GAMMA)


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


def visualize_state(state: np.ndarray, adhdp: ADHDP) -> None:
    a, adot = state
    x = 0
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

    rl.draw_text(f"R = {adhdp.reward_prev}", 10, 10, 12, HIGHLIGHT)
    rl.draw_text(f"Ec = {adhdp.error[1]}", 10, 20, 12, HIGHLIGHT)

    rl.end_drawing()


def cart_pole_test():

    input_dim = 2
    output_dim = 1
    actor = FFNN([
        Layer.linear(input_dim),
        #Layer.tanh(2),
        Layer.linear(output_dim),
    ], (-0.4, 0.4), (-0, 0))
    actor.weights = np.array([
        #1,0,0,
        #0,1,0,
        0.2, 0.1, 0.0  # Proportional gain, Derivative gain, bias
    ])
    
    c_w = 1
    critic = FFNN([
        Layer.linear(input_dim + output_dim),
        Layer.tanh(5),
        Layer.tanh(5),
        Layer.linear(1),
    ], (-c_w, c_w), (-2, 2))
    
    #critic = RBFNN.grid_spaced(output_dim,
    #    np.linspace(-0.7, 0.7, 5), 
    #    np.linspace(-0.7, 0.7, 5), 
    #    np.linspace(-0.7, 0.7, 5))
    
    #prep_critic(critic)
    #critic.save_to("saved_networks/pendulum/critic15tanh_prep.dat")
    critic.load_from("saved_networks/pendulum/critic5x5tanh_prep.dat")

    #critic.load_from("saved_networks/pendulum/critic_trained_p5.dat")
    #actor.load_from("saved_networks/pendulum/actor_trained_p5.dat")

    adhdp = ADHDP(actor, critic)

    if DISPLAY:
        #rl.set_target_fps(100)
        rl.init_window(800, 500, "Pendulum")
    
    epochs = 100

    error_evolution = np.zeros(epochs)

    window_closed = False

    for i in range(epochs):
        a = 0
        a_dot = np.random.normal(0, 2)
        state = np.array([a, a_dot])

        time = 0
        error_evolution_run = []
        adhdp.J_prev = adhdp.critic.eval(np.array([a, a_dot, 0]))

        while True:
            dt = TIMESTEP

            state = adhdp.step_and_learn(state, cart_pole_dynamics, cart_pole_reward, dt)

            if time > 10 or abs(state[1]) > 20 or adhdp.reward_prev > 1-1e-3:
                break

            # Visualization
            if DISPLAY:
                visualize_state(state, adhdp)
                if rl.window_should_close():
                    window_closed = True
                    break

            time += dt
            error_evolution_run.append(adhdp.error[1])

        if window_closed:
            break

        error_evolution[i] = np.average(np.array(error_evolution_run))
        print(i, np.average(np.array(error_evolution_run)), np.median(np.array(error_evolution_run)))
        
    
    if DISPLAY:
        rl.close_window()

    #critic.save_to("saved_networks/pendulum/critic_trained_p95.dat")
    #actor.save_to("saved_networks/pendulum/actor_trained_p95.dat")

    plt.plot(error_evolution, label="trained")
    #plt.ylim((1, 10))
    #plt.legend()
    plt.show()

    plot_critic(adhdp)

    if DISPLAY:
        rl.close_window()


def prep_critic(critic: Network):
    ''' Trains critic on utility function '''
    N = 2000
    ee = np.zeros(N)

    for i in range(N):
        x = np.random.uniform(-1, 1, 3)
        r = cart_pole_reward(x[:2] * np.array([pi, 20]))
        out, ee[i] = critic.learn_ml(0.01, x, r)
        #ee[i], _ = critic.learn_gd(1e-2, x, r)
        print(i, ee[i], x, out)

    plt.plot(ee)
    plt.show()
    
if __name__ == "__main__":
    cart_pole_test()
