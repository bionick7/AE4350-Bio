from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pyray as rl
from copy import copy

from common import *
from adhdp import ADHDP, ADHDPState
from networks import Network, Layer, FFNN, RBFNN, levenberg_marquardt

FPS: int = 100000000
TIMESTEP = 0.1
DISPLAY = True

class Acc1D(ADHDPState):
    force_multiplier = 10

    @classmethod
    def _dynamics(cls, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        ''' Also integrates dxdu '''

        p, p_dot = s
        F = u[0] * cls.force_multiplier

        p_ddot = F

        return np.array([p_dot, p_ddot])

    def step_forward(self, u: np.ndarray, dt: float) -> Acc1D:
        next_states = np.zeros(self.internal_state.shape)
        init_states = np.zeros(self.internal_state.shape)
        init_states[:,:2] = self.internal_state[:,:2]
        for i in range(len(self.internal_state)):
            next_states[i] = rk4(self._dynamics, init_states[i], u[i], dt)
        return Acc1D(next_states)

    def get_s(self) -> np.ndarray:
        return self.internal_state

    def get_dsdu(self, dt: float, u: np.ndarray) -> np.ndarray:
        dsdu_single = np.array([[self.force_multiplier*dt*dt*0.5], [self.force_multiplier*dt]])
        return np.repeat(dsdu_single[np.newaxis,:], len(self), axis=0)

    def get_reward(self) -> np.ndarray:
        p, p_dot, *_ = self.internal_state.T
        return (p*p + p_dot*p_dot) ** 0.1 * (1 - self.config["gamma"])

    def __copy__(self) -> ADHDPState:
        return ADHDPState(self.internal_state.copy())


class Acc2D(ADHDPState):
    force_multiplier = 10

    def step_forward(self, u: np.ndarray, dt: float) -> Acc2D:
        next_states = np.zeros(self.internal_state.shape)
        dsdu = self.get_dsdu(dt, u)
        for i in range(len(self)):
            next_states[i] = self.internal_state[i] + dsdu[i] @ u[i]  # exact, since it's linear
            next_states[i,:2] += self.internal_state[i,2:] * dt
        return Acc2D(next_states)

    def get_initial_control(self) -> np.ndarray:
        x, y, x_dot, y_dot = self.internal_state.T
        u = np.zeros((len(self), 2))
        u[:,0] = - x - x_dot
        u[:,1] = - y - y_dot * .5
        return u
    
    def get_s(self) -> np.ndarray:
        return self.internal_state

    def get_dsdu(self, dt: float, u: np.ndarray) -> np.ndarray:
        single_dsdu = np.vstack((
            np.eye(2) * dt*dt * .5,
            np.eye(2) * dt,
        )) * self.force_multiplier
        return np.repeat(single_dsdu[np.newaxis,:], len(self), axis=0)

    def get_reward(self, adhdp: ADHDP) -> np.ndarray:
        x, y, x_dot, y_dot = self.get_s().T
        return (x*x+y*y + x_dot*x_dot+y_dot*y_dot - 1) * (1 - self.config["gamma"])


def visualize_state(states: ADHDPState, adhdp: ADHDP) -> None:
    W, H = rl.get_screen_width(), rl.get_screen_height()

    mid_x = int(W*0.5)
    mid_y = int(H*0.5)

    rl.begin_drawing()
    rl.clear_background(BG)
    for ss in states.get_s():
        p, p_dot = ss
        rl.draw_circle(mid_x + int(p * 300), mid_y, 30, FG)
        rl.draw_circle(mid_x, mid_y, 10, HIGHLIGHT)
        rl.draw_line(mid_x + int(p * 300), mid_y,
                    mid_x + int((p + p_dot) * 300), mid_y, HIGHLIGHT)
        
        rl.draw_circle(mid_x + int(p * 300), mid_y + int(p_dot * 300), 10, rl.BLUE)
        rl.draw_rectangle_lines(mid_x-300, mid_y-300, 600, 600, rl.BLUE)

        #rl.draw_text(f"R = {adhdp.reward_prev}", 10, 10, 12, HIGHLIGHT)
        #rl.draw_text(f"Ec = {adhdp.error[1]}", 10, 20, 12, HIGHLIGHT)

    rl.end_drawing()


def visualize_state_2d(states: ADHDPState, adhdp: ADHDP) -> None:
    W, H = rl.get_screen_width(), rl.get_screen_height()

    mid_x = int(W*0.5)
    mid_y = int(H*0.5)

    rl.begin_drawing()
    rl.clear_background(BG)

    J = np.zeros(len(states))
    for i, ss in enumerate(states.get_s()):
        x, y, x_dot, y_dot = ss
        rl.draw_circle(mid_x + int(x * 300), mid_y + int(y * 300), 30, FG)
        rl.draw_line(  mid_x + int(x * 300), mid_y + int(y * 300),
                       mid_x + int((x + x_dot) * 300), mid_y + int((y + y_dot) * 300), 
                       HIGHLIGHT)
        J[i] = adhdp.critic.eval(np.concatenate((ss, adhdp.actor.eval(ss))))

    reward = np.average(states.get_reward())
    rl.draw_text(f"J = {np.average(J)}", 10, 10, 16, HIGHLIGHT)
    rl.draw_text(f"r = {reward}", 10, 30, 16, HIGHLIGHT)
    #rl.draw_text(f"Ec = {adhdp.error[1]}", 10, 20, 12, HIGHLIGHT)

    rl.draw_line(mid_x, 0, mid_x, H, HIGHLIGHT)
    rl.draw_line(0, mid_y, W, mid_y, HIGHLIGHT)
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
        -0.1,-0.1,0  # Proportional gain, Derivative gain, bias
    ], float)
    
    c_w = 1
    critic = FFNN([
        Layer.linear(input_dim + output_dim),
        Layer.tanh(5),
        Layer.tanh(5),
        Layer.linear(1),
    ], (-c_w, c_w), (-c_w, c_w))
    
    plant = FFNN([
        Layer.linear(input_dim + output_dim),
        Layer.tanh(5),
        Layer.tanh(5),
        Layer.linear(input_dim),
    ], (-c_w, c_w), (-c_w, c_w))
    
    #critic = RBFNN.grid_spaced(output_dim,
    #    np.linspace(-0.7, 0.7, 5), 
    #    np.linspace(-0.7, 0.7, 5), 
    #    np.linspace(-0.7, 0.7, 5))
    
    #prep_critic(critic)
    #critic.save_weights_to("acc1d/critic5x5tanh_prep.dat")
    #critic.save_weights_to("acc1d/critic15tanh_prep.dat")
    #critic.load_weights_from("acc1d/critic15tanh_prep.dat")
    #critic.load_weights_from("acc1d/critic5x5tanh_prep.dat")

    #critic.load_weights_from("acc1d/critic_trained_p5.dat")
    #actor.load_weights_from("acc1d/actor_trained_p5.dat")

    critic.load_weights_from("acc1d/critic_trained_p99.dat")
    critic.set_weight_matrix(0, critic.get_weight_matrix(0) + np.random.normal(0, 0.1))
    actor.load_weights_from("acc1d/actor_trained_p99.dat")

    adhdp = ADHDP(actor, critic, plant, population)

    if DISPLAY:
        rl.set_target_fps(FPS)
        rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
        rl.init_window(800, 700, "Pendulum")

    adhdp.gamma = .99
    adhdp.train_critic = True
    adhdp.train_actor = True
    adhdp.use_actor = True
    adhdp.actor_learning_rate = 1e-3
    adhdp.critic_learning_rate = 1e-2

    window_closed = False
    error_evolution = np.zeros((epochs, 3))
    for epoch in range(epochs):
        state_init = np.random.uniform(-1, 1, (population, 2))
        state = Acc1D(state_init)

        if adhdp.gamma < 0.99:
            adhdp.gamma = 1 - (1-adhdp.gamma) * 0.79

        time = 0
        error_evolution_run = []

        while True:
            #adhdp.u_offsets = np.random.normal(0, 0.01, (population, 1))

            dt = TIMESTEP

            state = adhdp.step_and_learn(state, dt)
            r = state.get_reward()

            if np.average(r) < 0.001:
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

        error_evolution[epoch] = np.average(np.average(np.array(error_evolution_run), axis=2), axis=0)
        print(epoch, error_evolution[epoch])
    
    if DISPLAY:
        rl.close_window()


    #critic.save_weights_to("acc1d/critic_trained_p99.dat")
    #actor.save_weights_to("acc1d/actor_trained_p99.dat")

    adhdp.plot_actor_critic()
    adhdp.plot_critic_gradient(0,1, 0,1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(error_evolution[:,0], label='actor')
    ax.plot(error_evolution[:,1], label='critic')
    fig.legend()

    plt.show()


def acceleration_test_2d():
    population = 60
    epochs = 20
    
    #actor = RBFNN.grid_spaced(2,
    #    np.linspace(-1, 1, 4),
    #    np.linspace(-1, 1, 4),
    #    np.linspace(-1, 1, 4),
    #    np.linspace(-1, 1, 4)
    #)
    actor = FFNN([
        Layer.linear(4),
        Layer.tanh(4),
        Layer.tanh(2),
    ])

    #actor.set_weight_matrix(0, np.hstack((np.eye(4), np.zeros((4, 1)))))
    #actor.set_weight_matrix(-1, np.hstack((-0.1 * np.eye(2), -0.1 * np.eye(2), np.zeros((2, 1)))))
    
    critic = FFNN([
        Layer.linear(6),
        Layer.tanh(20),
        Layer.tanh(20),
        Layer.linear(1)
    ])

    plant = FFNN([Layer.linear(6),Layer.linear(4)])

    adhdp = ADHDP(actor, critic, plant, population)

    if DISPLAY:
        rl.set_target_fps(FPS)
        rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
        rl.init_window(800, 700, "Acc2D")

    adhdp.gamma = 0.99
    adhdp.train_critic = True
    adhdp.train_actor = False
    adhdp.train_actor_on_initial = False
    adhdp.train_plant = False

    adhdp.use_plant = False
    adhdp.use_actor = True
    adhdp.actor_learning_rate = 1e-3
    adhdp.critic_learning_rate = 1e-2
    #adhdp.plant_learning_rate = 1e-3

    critic.load_weights_from("acc2d/critic_trained_p99.dat")
    #critic.load_weights_from("acc2d/critic_trained_p0.dat")
    actor.load_weights_from("acc2d/actor_trained_p99.dat")

    window_closed = False
    error_evolution = np.zeros((epochs, 3))
    for epoch in range(epochs):
        state_init = np.random.uniform(-1, 1, (population, 4))
        state = Acc2D(state_init)

        #if epoch > 5:
        #    adhdp.use_plant = True

        if adhdp.gamma < 0.99:
            adhdp.gamma = 1 - (1-adhdp.gamma) * 0.9

        time = 0
        error_evolution_run = []

        while True:
            dt = TIMESTEP

            state = adhdp.step_and_learn(state, dt)
            r = state.get_reward()

            if np.median(r) < -0.00999:
                break

            if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                break

            if np.average(state.get_reward()) > 1e3:
                break

            # Visualization
            if DISPLAY:
                visualize_state_2d(state, adhdp)
                if rl.window_should_close():
                    window_closed = True
                    break

            time += dt
            error_evolution_run.append(adhdp.error.copy())

        if window_closed:
            break

        if len(error_evolution_run) != 0:
            error_evolution[epoch] = np.average(np.average(np.array(error_evolution_run), axis=2), axis=0)
            print(epoch, error_evolution[epoch])
    
    if DISPLAY:
        rl.close_window()

    #critic.save_weights_to("acc2d/critic_trained_p0.dat")
    #critic.save_weights_to("acc2d/critic_trained_p99.dat")
    #actor.save_weights_to("acc2d/actor_trained_p99.dat")
    #plant.save_weights_to("acc2d/plant_trained_p99.dat")
    #plant.show_weights()


    fig, ax = plt.subplots(1, 1)
    ax.plot(error_evolution[:,0], label='actor')
    ax.plot(error_evolution[:,1], label='critic')
    ax.plot(error_evolution[:,2], label='plant')
    fig.legend()

    adhdp.plot_actor_critic(0,1)
    adhdp.plot_critic_gradient(0,1, 0,1)

    plt.show()


def prep_critic(critic: Network):
    ''' Trains critic on utility function '''
    N = 1000
    ee = np.zeros(N)

    for i in range(N):
        x = np.random.uniform(-1, 1, 3)
        r = np.linalg.norm(x[:2])
        out, ee[i] = critic.learn_ml(0.01, x, np.array([r]))
        #out, ee[i] = critic.learn_gd(1e-3, x, r)
        print(i, ee[i], x, out)

    plt.plot(ee)
    #plt.show()


if __name__ == "__main__":
    acceleration_test()
    #acceleration_test_2d()
