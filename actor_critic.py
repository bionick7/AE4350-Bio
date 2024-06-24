import numpy as np
import pyray as rl

from math import sin, cos, fmod, pi
from typing import Any, Callable

from common import *
from networks import Network, Layer, check_gradients

class ADHDP:
    def __init__(self, p_actor: Network, p_critic: Network) -> None:
        self.actor = p_actor
        self.critic = p_critic

        self.J_prev = 0
        self.gamma = 0.2

        self.memory: tuple = ()

    def set_reward_function(self, reward_func) -> None:
        self.reward_function = reward_func

    def learn_step_start(self, input: np.ndarray) -> np.ndarray:
        u, grad_actor = self.actor.get_gradients(input)
        u = u.flatten()
        outp_critic, grad_critic = self.critic.get_gradients(np.concatenate((input, u)))
        J = outp_critic[0,0]

        # Cache results for learn_step_end
        self.memory = grad_actor, grad_critic, J
        return u
    
    def learn_step_end(self, reward: float):
        grad_actor, grad_critic, J = self.memory

        J_tgt = 10

        e_a = np.zeros((2, 1))
        E_a = .5 * (e_a.T@e_a)**2
        e_c = np.array([[self.J_prev - (self.gamma * J - reward)]])
        E_c = .5 * (e_c.T@e_c)**2

        # update critic weights
        critic_weight_delta = (grad_critic.T @ e_c).flatten() * -0.01
        self.critic.update_weights(critic_weight_delta)

        actor_weight_delta = (e_a @ dJdu + e_a @ dJdx @ dxdu).T @ grad_actor
        self.actor.update_weights(critic_weight_delta)

        self.J_prev = J


def cart_pole_dynamics(state: np.ndarray, F: float) -> np.ndarray:
    g0 = 9.81
    mp = 10.0
    mc = 10.0
    Lp = 1.0

    x, a, x_dot, a_dot = state

    a_ddot = ((g0 * sin(a) - cos(a) * (F + mp * Lp * a_dot**2 * sin(a)) / (mp + mc)) /
            # ----------------------------------------------------------------------------
                        (Lp * (4/3 - mp/(mp+mc) * cos(a)**2)))

    x_ddot = (F + mp*Lp * (sin(a)*a_dot**2 - a_ddot*cos(a))) / (mp + mc)

    return np.array([x_dot, a_dot, x_ddot, a_ddot])


def cart_pole_test():

    input_dim = 2
    output_dim = 1
    actor = Network([
        Layer.linear(input_dim),
        Layer.tanh(10),
        Layer.tanh(output_dim),
    ], (-1, 1), (-1, 1))
    critic = Network([
        Layer.linear(input_dim + output_dim),
        Layer.tanh(10),
        Layer.tanh(1),
    ], (-1, 1), (-1, 1))
    adhdp = ADHDP(actor, critic)

    #check_gradients(actor)
    #check_gradients(critic)

    rl.init_window(800, 500, "Pendulum")

    state = np.zeros(4)
    state[1] = np.pi*0.9
    x, a, x_dot, a_dot = state

    while not rl.window_should_close():
        dt = rl.get_frame_time()
        #dt = 0.01
        F = adhdp.learn_step_start(np.array([a, x]))[0]
        F = -30*a - 2*a_dot - 50*x - 0*x_dot  # override with PD control to only train critic

        # Runge-Kutta
        k1 = cart_pole_dynamics(state, F)
        k2 = cart_pole_dynamics(state + k1*dt/2, F)
        k3 = cart_pole_dynamics(state + k2*dt/2, F)
        k4 = cart_pole_dynamics(state + k3*dt, F)
        state += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        state[1] = fmod(state[1] + pi, 2*pi) - pi
        x, a, x_dot, a_dot = state


        #print(x, x_dot, a, a_dot)

        adhdp.learn_step_end(-a**2 - x**2)

        # Visualization
        rl.begin_drawing()
        rl.clear_background(BG)
        W, H = rl.get_screen_width(), rl.get_screen_height()
        Lp = 1.0
        screen_x1 = int(W*(x*0.2+0.5))
        screen_y1 = int(H * 0.5)
        head_x = x + Lp * sin(a)
        head_y = Lp * cos(a)
        screen_x2 = int(W*(head_x*0.2+.5))
        screen_y2 = int(H * 0.5 - W*0.2*head_y)
        rl.draw_rectangle(screen_x1-50, screen_y1, 100, 50, FG)
        rl.draw_circle(screen_x2, screen_y2, 30, FG)
        rl.draw_line_ex(rl.Vector2(screen_x1, screen_y1), rl.Vector2(screen_x2, screen_y2), 10, FG)
        rl.end_drawing()

    rl.close_window()

    
if __name__ == "__main__":
    cart_pole_test()
