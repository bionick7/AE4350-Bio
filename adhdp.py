import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Self
from copy import copy

from common import *
from networks import Network, Layer, FFNN, RBFNN, levenberg_marquardt

class ADHDPState:
    def __init__(self, p_internal_state: np.ndarray):
        self.internal_state = p_internal_state

    def step_forward(self, u: np.ndarray, dt: float) -> Self:
        res = copy(self)
        return res

    def get_x(self) -> np.ndarray:
        return self.internal_state

    def get_dxdu(self) -> np.ndarray:
        return np.zeros(len(self.internal_state))

    def get_reward(self, adhdhp: 'ADHDP') -> np.ndarray:
        return np.zeros(len(self))

    def __copy__(self) -> Self:
        return ADHDPState(self.internal_state.copy())
    
    def __len__(self) -> int:
        return len(self.internal_state)


class ADHDP:
    def __init__(self, p_actor: Network, p_critic: Network, p_plant: Network, population: int) -> None:
        self.actor = p_actor
        self.critic = p_critic
        self.plant = p_plant

        self.train_actor: bool = True
        self.train_critic: bool = True
        self.train_plant: bool = True
        self.use_plant: bool = False

        self.actor_learning_rate: float = 1e-3
        self.critic_learning_rate: float = 1e-3
        self.plant_learning_rate: float = 1e-3

        self.gamma: float = 0

        self.J_prev = np.zeros(population)
        self.reward_prev = np.zeros(population)
        self.error = np.zeros((3, population))

    def step_and_learn(self, states: ADHDPState, dt: float) -> ADHDPState:
        assert len(states) == len(self.J_prev)

        actor_inputs = states.get_x()
        dxdu = states.get_dxdu()

        x_size = self.actor.get_layer_size(0)
        u_size = self.actor.get_layer_size(-1)

        u = np.zeros((len(states), u_size))
        grad_actor = np.zeros((len(states), u_size, len(self.actor.weights)))
        
        for i in range(len(states)):
            u[i], grad_actor[i] = self.actor.get_weight_gradient(actor_inputs[i])
            #u[i] = self.get_optimal_actor_play(actor_inputs[i], dxdu[i])

        next_states = states.step_forward(u, dt)
        next_actor_inputs = next_states.get_x()
        next_dxdu = next_states.get_dxdu()
        rewards = next_states.get_reward(self)

        for i in range(len(states)):
            reward = rewards[i]
            critic_input = np.concatenate((next_actor_inputs[i], u[i]))

            if self.train_plant:
                _, E_p = self.plant.learn_gd(self.plant_learning_rate, critic_input, next_actor_inputs[i] - actor_inputs[i])
                #_, E_p = self.plant.learn_ml(0.001, critic_input, next_actor_inputs[i] - actor_inputs[i])
            else:
                #plant_out = self.plant.eval(critic_input)
                #target = next_actor_inputs[i] - actor_inputs[i]
                #e_p = target - plant_out
                #E_p = 0.5 * np.dot(e_p, e_p)
                E_p = 0


            # get reward
            critic_expected = self.gamma * self.J_prev[i] + reward  # previous input
            if self.train_critic:
                #J, E_c = self.critic.learn_ml(0.5, critic_input, critic_expected)
                J, E_c = self.critic.learn_gd(self.critic_learning_rate, critic_input, critic_expected)
                J = J[0]
            else:
                J = self.critic.eval(critic_input)[0]
                E_c = (J - critic_expected)**2 * .5

            J_tgt = 0
            #J = rewards[i]
            e_a = np.array([[J - J_tgt]])
            E_a: float = .5 * np.dot(e_a, e_a)[0]

            self.error[:,i] = J, E_c, E_p
            
            if self.train_actor:
                # update actor weights
                critic_io_derivatives = self.critic.get_io_gradient(critic_input)
                dJdx = critic_io_derivatives[:,:x_size]
                dJdu = critic_io_derivatives[:,x_size:]
                if self.use_plant:
                    dxdu = self.plant.get_io_gradient(critic_input)[:,x_size:]
                else:
                    dxdu = next_dxdu[i].reshape(x_size, -1)

                true_grad_actor = (dJdu + dJdx @ dxdu) @ grad_actor[i]
                #true_grad_actor = dJdu @ grad_actor[i]

                actor_weight_delta = true_grad_actor.flatten() * -self.actor_learning_rate
                #actor_weight_delta = (true_grad_actor.T @ e_a).flatten() * -self.actor_learning_rate
                #actor_weight_delta = levenberg_marquardt(true_grad_actor, e_a, 0.01)
                self.actor.update_weights(actor_weight_delta)

            self.reward_prev[i] = reward
            self.J_prev[i] = J

        return next_states

    def get_optimal_actor_play(self, x: np.ndarray, dxdu: np.ndarray) -> np.ndarray:
        x_size = self.actor.get_layer_size( 0)
        u_size = self.actor.get_layer_size(-1)
        
        u = np.zeros(2)
        for i in range(10):
            critic_io_derivatives = self.critic.get_io_gradient(np.concatenate((x, u)))
            dJdx = critic_io_derivatives[:,:x_size]
            dJdu = critic_io_derivatives[:,x_size:]
            dJdu_total = (dJdu + dJdx @ dxdu).flatten()
            #dJdu_total = (dJdu).flatten()
            u -= dJdu_total * 0.5
        
        return u

    def reinitialize(self, state: ADHDPState) -> None:
        x_init = state.get_x()
        for i in range(len(state)):
            self.J_prev[i] = self.critic.eval(np.concatenate(((x_init[i], np.zeros(2)))))

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
                act_outp = self.actor.eval(actor_input)
                zz_actor[i, j] = act_outp[0]
                zz_critic[i, j] = self.critic.eval(np.concatenate((actor_input, act_outp)))[0]
        
        c1 = ax1.contourf(xx, yy, zz_critic)
        c2 = ax2.contourf(xx, yy, zz_actor)
        fig.colorbar(c1)
        fig.colorbar(c2)
        fig.tight_layout()
