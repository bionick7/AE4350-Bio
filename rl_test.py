import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from math import sin, cos, fmod, pi, inf, isinf
from typing import Any, Callable

from common import *
from networks import Network, Layer, FFNN, RBFNN, levenberg_marquardt

TIMESTEP = 0.1
DISPLAY = True

class ADHDP:
    def __init__(self, p_actor: Network, p_critic: Network, population: int) -> None:
        self.actor = p_actor
        self.critic = p_critic

        self.train_actor: bool = True
        self.train_critic: bool = True

        self.J_prev = np.zeros(population)
        self.reward_prev = np.zeros(population)
        self.gamma: float = 0
        self.error = [0, 0]

    def step_and_learn(self, states: np.ndarray, f_dynamcis: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                       f_reward: Callable[[np.ndarray], float], dt: float) -> np.ndarray:
        
        res = np.zeros(states.shape)
        for i, state in enumerate(states):
            u_raw, grad_actor = self.actor.get_weight_gradient(state)
            u_raw = u_raw.flatten()

            u = u_raw + np.random.normal(0, 0.05)

            # Runge-Kutta
            x_size = self.actor.get_layer_size(0)
            integration_state = np.concatenate((state, state*0))
            k1 = f_dynamcis(integration_state, u)
            k2 = f_dynamcis(integration_state + k1*dt/2, u)
            k3 = f_dynamcis(integration_state + k2*dt/2, u)
            k4 = f_dynamcis(integration_state + k3*dt, u)
            prev_integration_state = integration_state
            integration_state += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            integration_state[0] = fmod(integration_state[0] + 1, 2) - 1

            reward = f_reward(integration_state[:x_size])
            critic_input = np.concatenate((integration_state[:x_size], u_raw))
             
            # get reward
            critic_expected = self.gamma * self.J_prev[i] + f_reward(state)  # previous input
            if self.train_critic:
                #J, E_c = self.critic.learn_ml(0.01, critic_input, critic_expected)
                J, E_c = self.critic.learn_gd(1e-3, critic_input, critic_expected)
            else:
                J = self.critic.eval(critic_input).flatten()[0]
                E_c = (J - critic_expected)**2 * .5

            J_tgt = 0
            e_a = np.array([[J - J_tgt]])
            E_a: float = .5 * (e_a.T@e_a).flatten()[0]

            self.error = [E_a, E_c]
            
            if self.train_actor:
                # update actor weights
                critic_io_derivatives = self.critic.get_io_gradient(critic_input)
                dJdx = critic_io_derivatives[:,:x_size]
                dJdu = critic_io_derivatives[:,x_size:]
                dxdu = prev_integration_state[x_size:]

                true_grad_actor = (dJdu + dJdx @ dxdu).T @ grad_actor

                actor_weight_delta = true_grad_actor.T.flatten() * -1e-3
                #actor_weight_delta = levenberg_marquardt(true_grad_actor, e_a, 0.2)
                self.actor.update_weights(actor_weight_delta)

            self.reward_prev[i] = reward
            self.J_prev[i] = J
            res[i] = integration_state[:x_size]

        return res

    def plot_actor_critic(self):
        fig = plt.figure(figsize=(9, 4))
        ax1, ax2 = fig.subplots(1, 2)
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        zz = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                inp = np.array([xx[i, j], yy[i, j]])
                u = self.actor.eval(inp).flatten()
                zz[i, j] = self.critic.eval(np.concatenate((inp, u)))[0,0]
                #zz[i, j] = acceleration_reward(inp)
        #plt.axis('scaled')
        #plt.contourf(xx, yy, zz, levels=np.linspace(0,1,10), extend='both')
        c1 = ax1.contourf(xx, yy, zz)
        fig.colorbar(c1)

        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        zz = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = self.actor.eval(np.array([xx[i, j], yy[i, j]]))[0,0]
        #plt.axis('scaled')
        c2 = ax2.contourf(xx, yy, zz)
        fig.colorbar(c2)
        fig.tight_layout()


def acceleration_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    ''' Also integrates dxdu '''
    force_multiplier = 10

    p, p_dot, dp_du, dpdot_du = x
    F = u[0] * force_multiplier

    p_ddot = F
    dpddot_du = force_multiplier

    return np.array([p_dot, p_ddot, dpdot_du, dpddot_du])


def acceleration_reward(x: np.ndarray, gamma: float) -> float:
    p, p_dot = x
    return np.sqrt(p**2 + p_dot**2) * (1 - gamma)


def visualize_state(states: np.ndarray, adhdp: ADHDP) -> None:
    W, H = rl.get_screen_width(), rl.get_screen_height()

    mid_x = int(W*0.5)
    mid_y = int(H*0.5)

    rl.begin_drawing()
    rl.clear_background(BG)
    for state in states:
        p, p_dot = state
        rl.draw_circle(mid_x + int(p * 300), mid_y, 30, FG)
        rl.draw_circle(mid_x, mid_y, 10, HIGHLIGHT)
        rl.draw_line(mid_x + int(p * 300), mid_y,
                    mid_x + int((p + p_dot) * 300), mid_y, HIGHLIGHT)
        
        rl.draw_circle(mid_x + int(p * 300), mid_y + int(p_dot * 300), 10, rl.BLUE)
        rl.draw_rectangle_lines(mid_x-300, mid_y-300, 600, 600, rl.BLUE)

        #rl.draw_text(f"R = {adhdp.reward_prev}", 10, 10, 12, HIGHLIGHT)
        #rl.draw_text(f"Ec = {adhdp.error[1]}", 10, 20, 12, HIGHLIGHT)

    rl.end_drawing()


def acceleration_test():
    population = 10
    epochs = 30

    input_dim = 2
    output_dim = 1
    actor = FFNN([
        Layer.linear(input_dim),
        Layer.tanh(2),
        Layer.linear(output_dim),
    ], (-1, 1), (-1, 1))
    actor.weights = np.array([
        1,0,0,
        0,1,0,
        -0.5,-0.5,0  # Proportional gain, Derivative gain, bias
    ], float)
    
    c_w = 1
    critic = FFNN([
        Layer.linear(input_dim + output_dim),
        Layer.tanh(5),
        Layer.tanh(5),
        Layer.linear(1),
    ], (-c_w, c_w), (-c_w, c_w))
    
    #critic = RBFNN.grid_spaced(output_dim,
    #    np.linspace(-0.7, 0.7, 5), 
    #    np.linspace(-0.7, 0.7, 5), 
    #    np.linspace(-0.7, 0.7, 5))
    
    #prep_critic(critic)
    #critic.save_to("saved_networks/pendulum/critic5x5tanh_prep.dat")
    #critic.save_to("saved_networks/pendulum/critic15tanh_prep.dat")
    #critic.load_from("saved_networks/pendulum/critic15tanh_prep.dat")
    #critic.load_from("saved_networks/pendulum/critic5x5tanh_prep.dat")

    #critic.load_from("saved_networks/pendulum/critic_trained_p5.dat")
    #actor.load_from("saved_networks/pendulum/actor_trained_p5.dat")

    critic.load_from("saved_networks/pendulum/critic_trained_p99.dat")
    #actor.load_from("saved_networks/pendulum/actor_trained_p99.dat")

    adhdp = ADHDP(actor, critic, population)

    if DISPLAY:
        #rl.set_target_fps(10)
        rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
        rl.init_window(800, 700, "Pendulum")

    adhdp.gamma = .99
    adhdp.train_critic = True
    adhdp.train_actor = True

    window_closed = False
    error_evolution = np.zeros((epochs, 2))
    for epoch in range(epochs):
        state = np.zeros((population, 2))
        state[:,0] = np.random.uniform(-1, 1, population)
        state[:,1] = np.random.uniform(-1, 1, population)

        if adhdp.gamma < 0.99:
            adhdp.gamma = 1 - (1-adhdp.gamma) * 0.79

        time = 0
        error_evolution_run = []
        for i in range(population):
            adhdp.J_prev[i] = adhdp.critic.eval(np.append(state[i], 0))

        while True:
            dt = TIMESTEP

            state = adhdp.step_and_learn(state, acceleration_dynamics, 
                                         lambda x: acceleration_reward(x, adhdp.gamma), dt)

            if time > 10:
                break

            # Visualization
            if DISPLAY:
                visualize_state(state, adhdp)
                if rl.window_should_close():
                    window_closed = True
                    break

            time += dt
            error_evolution_run.append(adhdp.error)

        if window_closed:
            break

        error_evolution[epoch] = np.average(np.array(error_evolution_run), axis=0)
        print(epoch, error_evolution[epoch])
    
    if DISPLAY:
        rl.close_window()


    critic.save_to("saved_networks/pendulum/critic_trained_p99.dat")
    actor.save_to("saved_networks/pendulum/actor_trained_p99.dat")

    adhdp.plot_actor_critic()

    fig, ax = plt.subplots(1, 1)
    ax.plot(error_evolution[:,0], label='actor')
    ax.plot(error_evolution[:,1], label='critic')
    fig.legend()

    plt.show()
    


def prep_critic(critic: Network):
    ''' Trains critic on utility function '''
    N = 1000
    ee = np.zeros(N)

    for i in range(N):
        x = np.random.uniform(-1, 1, 3)
        r = acceleration_reward(x[:2], 0)
        out, ee[i] = critic.learn_ml(0.01, x, r)
        #out, ee[i] = critic.learn_gd(1e-3, x, r)
        print(i, ee[i], x, out)

    plt.plot(ee)
    plt.show()
    
if __name__ == "__main__":
    acceleration_test()
