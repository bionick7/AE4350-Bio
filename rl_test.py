import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from math import sin, cos, fmod, pi, inf, isinf
from typing import Any, Callable

from common import *
from networks import Network, Layer, FFNN, RBFNN, check_gradients


class ADHDP:
    def __init__(self, p_actor: Network, p_critic: Network) -> None:
        self.actor = p_actor
        self.critic = p_critic

        self.J_prev = 0
        self.gamma = 0.0
        self.error = [0, 0]

        self.memory: tuple = ()

    def learn_step_start(self, input: np.ndarray, u_over: float=inf) -> np.ndarray:
        u, grad_actor = self.actor.get_gradients(input)
        u = u.flatten()
        if isinf(u_over):
            outp_critic, grad_critic = self.critic.get_gradients(np.concatenate((input, u)))
        else:
            outp_critic, grad_critic = self.critic.get_gradients(np.append(input, u_over))
        J = outp_critic[0,0]

        # Cache results for learn_step_end
        self.memory = grad_actor, grad_critic, J
        return u
    
    def learn_step_end(self, reward: float, initial_timestep: bool=False):
        grad_actor, grad_critic, J = self.memory

        J_tgt = 0

        e_a = np.zeros((2, 1))
        E_a: float = .5 * (e_a.T@e_a).flatten()[0]
        e_c = np.array([[self.J_prev - (self.gamma * J + reward)]])
        if initial_timestep:
            e_c *= 0
        E_c: float = .5 * (e_c.T@e_c).flatten()[0]

        self.error = [E_a, E_c]

        # update critic weights
        critic_weight_delta = (grad_critic.T @ e_c).flatten() * -1e-2
        #critic_weight_delta = -np.linalg.solve(grad_critic.T@grad_critic - 0.2 * np.eye(grad_critic.shape[1]), grad_critic.T*e_c).flatten()
        self.critic.update_weights(critic_weight_delta)

        #actor_weight_delta = (e_a @ dJdu + e_a @ dJdx @ dxdu).T @ grad_actor
        #self.actor.update_weights(critic_weight_delta)

        #print(J, reward, E_c)

        self.J_prev = J


def cart_pole_dynamics(state: np.ndarray, F: float) -> np.ndarray:
    g0 = 9.81
    mp = 1.0
    mc = 10.0
    Lp = 1.0

    x, a, x_dot, a_dot = state

    a_ddot = ((g0 * sin(a) - cos(a) * (F + mp * Lp * a_dot**2 * sin(a)) / (mp + mc)) /
            # ----------------------------------------------------------------------------
                        (Lp * (4/3 - mp/(mp+mc) * cos(a)**2)))

    x_ddot = (F + mp*Lp * (sin(a)*a_dot**2 - a_ddot*cos(a))) / (mp + mc)

    return np.array([x_dot, a_dot, x_ddot, a_ddot])


def plot_critic(critic: Network):
    xx, yy = np.meshgrid(np.linspace(-pi, pi, 20), np.linspace(-1, 1, 20))
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = critic.eval(np.array([xx[i, j], -0.5, yy[i, j]]))[0,0]
    #plt.axis('scaled')
    plt.contourf(xx, yy, zz)
    plt.colorbar()
    plt.show()
    

def cart_pole_test():

    input_dim = 2
    output_dim = 1
    actor = FFNN([
        Layer.linear(input_dim),
        Layer.linear(output_dim),
    ], (-1, 1), (-1, 1))
    #actor.weights = np.array([0.2, 0.1, 0.0])  # Proportional gain, Derivative gain, bias
    c_w = 1
    critic = FFNN([
        Layer.linear(input_dim + output_dim),
        Layer.tanh(20),
        Layer.linear(1),
    ], (-c_w, c_w), (-c_w, c_w))
    ref_critic = FFNN(critic.architecture, (critic.w_min, critic.w_max), (critic.b_min, critic.b_max))
    
    #critic_rbfs = 200
    #critic = RBFNN(input_dim + output_dim, critic_rbfs, output_dim, (-c_w, c_w), (-c_w, c_w))
    #centers = np.hstack((
    #    np.random.uniform(-np.pi, np.pi, (critic_rbfs, 1)),
    #    np.random.uniform(-np.pi, np.pi, (critic_rbfs, 1)),
    #    np.random.uniform(-1, 1, (critic_rbfs, 1))
    #))
    #critic.set_rbfs(centers, np.ones(centers.shape) * 0.5)
    #ref_critic = RBFNN(input_dim + output_dim, critic_rbfs, output_dim, (critic.w_min, critic.w_max), (critic.b_min, critic.b_max))
    #ref_critic.weights = critic.weights.copy()
    #ref_critic.set_rbfs(critic.centers, critic.stdevs)
    
    adhdp = ADHDP(actor, critic)

    #check_gradients(actor)
    #check_gradients(critic)

    #rl.set_target_fps(10)
    #rl.init_window(800, 500, "Pendulum")
    
    error_evolution = np.zeros(100)
    ref_error_evolution = np.zeros(100)

    for i in range(100):
        state = np.zeros(4)
        state[1] = np.random.uniform(-np.pi, np.pi)
        state[3] = np.random.normal(0, 0.1)

        time = 0
        ref_J_prev = 0
        print(i)
        error_evolution_run = []
        ref_error_evolution_run = []

        while True:
            x, a, x_dot, a_dot = state
            dt = 0.1#rl.get_frame_time()

            #F = 200*a + 100*a_dot# - 10*x - 10*x_dot  # override with PD control to only train critic
            #F += np.random.uniform(0, 100)
            u = adhdp.learn_step_start(np.array([a, a_dot]))[0]
            F = u * 1000

            # Runge-Kutta
            k1 = cart_pole_dynamics(state, F)
            k2 = cart_pole_dynamics(state + k1*dt/2, F)
            k3 = cart_pole_dynamics(state + k2*dt/2, F)
            k4 = cart_pole_dynamics(state + k3*dt, F)
            state += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            state[1] = fmod(state[1] + pi, 2*pi) - pi
            #state[0] = max(-1, min(1, state[0]))
            x, a, x_dot, a_dot = state

            #print(x, x_dot, a, a_dot)

            #adhdp.learn_step_end(a**2 + np.log(x**2+1))
            reward = a**2 + a_dot**2
            adhdp.learn_step_end(reward, time==0)
            if reward < 1e-5 or time > 100:
                break

            # Visualization
            #rl.begin_drawing()
            #rl.clear_background(BG)
            W, H = rl.get_screen_width(), rl.get_screen_height()
            Lp = 1.0
            screen_x1 = int(W*(x*0.2+0.5))
            offset = screen_x1 - int(W*0.5)
            screen_y1 = int(H * 0.5)
            head_x = x + Lp * sin(a)
            head_y = Lp * cos(a)
            screen_x2 = int(W*(head_x*0.2+.5))
            screen_y2 = int(H * 0.5 - W*0.2*head_y)
            #rl.draw_rectangle(screen_x1-50-offset, screen_y1, 100, 50, FG)
            #rl.draw_circle(screen_x2-offset, screen_y2, 30, FG)
            #rl.draw_line_ex(rl.Vector2(screen_x1-offset, screen_y1), rl.Vector2(screen_x2-offset, screen_y2), 10, FG)
            #rl.draw_rectangle(screen_x1-50, screen_y1, 100, 50, FG)
            #rl.draw_circle(screen_x2, screen_y2, 30, FG)
            #rl.draw_line_ex(rl.Vector2(screen_x1, screen_y1), rl.Vector2(screen_x2, screen_y2), 10, FG)
            #rl.end_drawing()
            time += dt
            error_evolution_run.append(adhdp.error[1])
            ref_J = ref_critic.eval(np.array([a, a_dot, u]))[0]
            ref_e_c = ref_J_prev - (adhdp.gamma * ref_J + reward)
            ref_error_evolution_run.append(0.5 * ref_e_c**2)
            ref_J_prev = ref_J

        error_evolution[i] = np.mean(np.array(error_evolution_run))
        ref_error_evolution[i] = np.mean(np.array(ref_error_evolution_run))

    #rl.close_window()
    plt.semilogy(ref_error_evolution + 1, label="reference")
    plt.semilogy(error_evolution + 1, label="trained")
    plt.ylim((1, 10))
    plt.legend()
    plt.show()

    plot_critic(critic)
    
    
if __name__ == "__main__":
    cart_pole_test()
