from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from copy import copy

from common import *
from networks import Network, Layer, FFNN, RBFNN, levenberg_marquardt

class ADHDPState:
    def __init__(self, p_internal_state: np.ndarray):
        self.internal_state = p_internal_state

    def step_forward(self, u: np.ndarray, dt: float) -> ADHDPState:
        res = copy(self)
        return res

    def get_x(self) -> np.ndarray:
        return self.internal_state

    def get_dxdu(self) -> np.ndarray:
        return np.zeros(len(self.internal_state))

    def get_reward(self) -> np.ndarray:
        return np.zeros(len(self))

    def __copy__(self) -> ADHDPState:
        return ADHDPState(self.internal_state.copy())
    
    def __len__(self) -> int:
        return len(self.internal_state)


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

    def step_and_learn(self, states: ADHDPState, dt: float) -> ADHDPState:
        assert len(states) == len(self.J_prev)

        actor_inputs = states.get_x()

        x_size = self.actor.get_layer_size(0)
        u_size = self.actor.get_layer_size(-1)

        u = np.zeros((len(states), u_size))
        grad_actor = np.zeros((len(states), u_size, len(self.actor.weights)))

        for i in range(len(states)):
            u[i], grad_actor[i] = self.actor.get_weight_gradient(actor_inputs[i])

        next_states = states.step_forward(u, dt)
        next_actor_inputs = next_states.get_x()
        next_dxdu = next_states.get_dxdu()
        rewards = next_states.get_reward()

        for i in range(len(states)):
            reward = rewards[i]
            critic_input = np.concatenate((next_actor_inputs[i], u[i]))
             
            # get reward
            critic_expected = self.gamma * self.J_prev[i] + reward  # previous input
            if self.train_critic:
                #J, E_c = self.critic.learn_ml(0.01, critic_input, critic_expected)
                J, E_c = self.critic.learn_gd(1e-3, critic_input, critic_expected)
                J = J[0]
            else:
                J = self.critic.eval(critic_input)[0]
                E_c = (J - critic_expected)**2 * .5

            J_tgt = 0
            e_a = np.array([[J - J_tgt]])
            E_a: float = .5 * np.dot(e_a, e_a)[0]

            self.error = [J, E_c]
            
            if self.train_actor:
                # update actor weights
                critic_io_derivatives = self.critic.get_io_gradient(critic_input)
                dJdx = critic_io_derivatives[:,:x_size]
                dJdu = critic_io_derivatives[:,x_size:]
                dxdu = next_dxdu[i].reshape(x_size, -1)

                true_grad_actor = (dJdu + dJdx @ dxdu) @ grad_actor[i]
                #true_grad_actor = dJdu @ grad_actor[i]

                actor_weight_delta = true_grad_actor.flatten() * -1e-4
                #actor_weight_delta = (true_grad_actor.T @ e_a).flatten() * -1e-3
                #actor_weight_delta = levenberg_marquardt(true_grad_actor, e_a, 0.01)
                self.actor.update_weights(actor_weight_delta)

            self.reward_prev[i] = reward
            self.J_prev[i] = J

        return next_states

    def plot_actor_critic(self, axis1: int=0, axis2: int=1):
        fig = plt.figure(figsize=(9, 4))
        ax1, ax2 = fig.subplots(1, 2)
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        zz_actor = np.zeros(xx.shape)
        zz_critic = np.zeros(xx.shape)
        actor_input = np.zeros(self.actor.get_layer_size(0))

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                actor_input[axis1] = xx[i, j]
                actor_input[axis2] = yy[i, j]
                zz_actor[i, j] = self.actor.eval(actor_input).flatten()
                zz_critic[i, j] = self.critic.eval(np.concatenate((actor_input, zz_actor[i, j])))[0]
        
        c1 = ax1.contourf(xx, yy, zz_critic)
        c2 = ax2.contourf(xx, yy, zz_actor)
        fig.colorbar(c1)
        fig.colorbar(c2)
        fig.tight_layout()
