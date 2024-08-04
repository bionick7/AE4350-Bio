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

    def get_initial_control(self) -> np.ndarray:
        return np.zeros((len(self.internal_state), 1))

    def get_s(self) -> np.ndarray:
        return self.internal_state

    def get_dsdu(self, dt: float, u: np.ndarray) -> np.ndarray:
        return np.zeros(len(self.internal_state))
    
    def check_dsdu(self, u_size: int):
        EPSILON_U = 0.1
        DELTA_T = 0.1
        u = np.zeros((len(self), u_size))
        dsdu = self.get_dsdu(DELTA_T, u)
        dsdu_fd = np.zeros(dsdu.shape)
        dpdu_fd = np.zeros((len(self), self.internal_state.shape[1], dsdu.shape[2]))
        for u_axis in range(dsdu.shape[2]):
            u[:,u_axis] = EPSILON_U
            state_plus = self.step_forward(u, DELTA_T)
            u[:,u_axis] = -EPSILON_U
            state_minus = self.step_forward(u, DELTA_T)
            dsdu_fd[:,:,u_axis] = (state_plus.get_s() - state_minus.get_s()) / (2*EPSILON_U)
            dpdu_fd[:,:,u_axis] = (state_plus.internal_state - state_minus.internal_state) / (2*EPSILON_U)
            
        print(dsdu_fd / dsdu)

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

        self.train_actor: bool = False
        self.train_actor_on_initial: bool = False
        self.train_critic: bool = False
        self.train_plant: bool = False

        self.use_plant: bool = True
        self.use_actor: bool = True

        self.actor_learning_rate: float = 1e-3
        self.critic_learning_rate: float = 1e-3
        self.plant_learning_rate: float = 1e-3

        self.actor_weight_mask = np.ones(self.actor.weights.shape)

        self.gamma: float = 0

        self.error = np.zeros((3, population))

        self.s_size = self.actor.get_layer_size(0)
        self.u_size = self.actor.get_layer_size(-1)

        self.u = np.zeros((population, self.u_size))

        self.u_offsets = np.zeros((population, self.u_size))

        self.actor_save_path: str|None=None
        self.critic_save_path: str|None=None
        self.plant_save_path: str|None=None
       

    def step_and_learn(self, states: ADHDPState, dt: float) -> ADHDPState:
        actor_inputs = states.get_s()
        self.u = np.zeros((len(states), self.u_size))
        dsdu = states.get_dsdu(dt, self.u)

        #grad_actor = np.zeros((len(states), self.u_size, len(self.actor.weights)))
        
        if self.use_actor:
            for i in range(len(states)):
                self.u[i] = self.actor.eval(actor_inputs[i])
        else:
            self.u = states.get_initial_control()
            #for i in range(len(states)):
            #    u[i] = self.get_optimal_actor_play(actor_inputs[i], dsdu[i])

        self.u += self.u_offsets

        next_states = states.step_forward(self.u, dt)
        next_actor_inputs = next_states.get_s()
        next_dsdu = next_states.get_dsdu(dt, self.u)
        rewards = states.get_reward(self)

        if self.train_actor_on_initial:
            u_expected = next_states.get_initial_control()

        for i in range(len(states)):
            reward = rewards[i]
            critic_input = np.concatenate((actor_inputs[i], self.u[i]))
            next_critic_input = np.concatenate((next_actor_inputs[i], self.u[i]))

            if self.train_plant:
                _, E_p = self.plant.learn_gd(self.plant_learning_rate, critic_input, next_actor_inputs[i] - actor_inputs[i])
                #_, E_p = self.plant.learn_ml(0.1, critic_input, next_actor_inputs[i] - actor_inputs[i])
            else:
                #plant_out = self.plant.eval(critic_input)
                #target = next_actor_inputs[i] - actor_inputs[i]
                #e_p = target - plant_out
                #E_p = 0.5 * np.dot(e_p, e_p)
                E_p = 0

            #     J(t+1) = gamma * J(t) + r(t+1)
            #     J(t) += r(t+1) + gamma * J(t+1) - J(t)
            J_next = self.critic.eval(next_critic_input)[0]
            critic_expected = self.gamma * J_next + reward  # previous input
            #critic_expected = (J_t1 - reward) / self.gamma  # previous input
            if self.train_critic:
                J_t, E_c = self.critic.learn_gd(self.critic_learning_rate, critic_input, critic_expected)
                #J_t, E_c = self.critic.learn_ml(0.1, critic_input, critic_expected)
            else:
                J_t = self.critic.eval(critic_input)
                E_c = (J_t[0] - critic_expected)**2 * .5

            J_tgt = 0
            E_a: float = 0
            
            if self.train_actor:
                #J = rewards[i]
                e_a = np.array([[J_next - J_tgt]])
                u_next, grad_actor = self.actor.get_weight_gradient(actor_inputs[i])

                # update actor weights
                dJdu = self.critic.get_io_gradient(critic_input)[:,self.s_size:]
                dJds = self.critic.get_io_gradient(next_critic_input)[:,:self.s_size]
                if self.use_plant:
                    dsdu_ = self.plant.get_io_gradient(critic_input)[:,self.s_size:]
                else:
                    dsdu_ = dsdu[i].reshape(self.s_size, self.u_size)

                grad_actor_now =  dJdu @ grad_actor
                grad_actor_next = dJds @ dsdu_ @ grad_actor
                true_grad_actor = grad_actor_now + grad_actor_next

                true_grad_actor *= self.actor_weight_mask

                actor_weight_delta = true_grad_actor.flatten() * -self.actor_learning_rate
                #actor_weight_delta = (true_grad_actor.T @ e_a).flatten() * -self.actor_learning_rate
                #actor_weight_delta = levenberg_marquardt(true_grad_actor, e_a, 0.01)
                self.actor.update_weights(actor_weight_delta)
                E_a: float = J_next
            
            elif self.train_actor_on_initial:
                _, E_a = self.actor.learn_gd(self.actor_learning_rate, next_actor_inputs[i], u_expected[i])

            self.error[:,i] = E_a, E_c, E_p

        return next_states

    def get_optimal_actor_play(self, s: np.ndarray, dsdu: np.ndarray) -> np.ndarray:
        u = np.zeros(2)
        critic_io_derivatives = self.critic.get_io_gradient(np.concatenate((s, u)))
        dJds = critic_io_derivatives[:,:self.s_size]
        dJdu = critic_io_derivatives[:,self.s_size:]
        dJdu_total = (dJdu + dJds @ dsdu).flatten()
        u -= dJdu_total

        u_norm = np.linalg.norm(u)
        if u_norm > 1:
            u /= u_norm
        
        return u

    def plot_actor_critic(self, axis1: int=0, axis2: int=1):
        fig = plt.figure(figsize=(9, 4))
        ax1, ax2 = fig.subplots(1, 2)
        xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
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

    def plot_critic_gradient(self, s_axis1: int=0, s_axis2: int=1, 
                             u_axis1: int=0, u_axis2: int=1, u_axis3: int=2):
        
        shown_u_size = min(self.u_size, 3)

        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.subplots(1, 1)
        xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        actor_input = np.zeros(self.actor.get_layer_size(0))
        critic_gradient = np.zeros((xx.shape[0], xx.shape[1], 3))

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                actor_input[s_axis1] = xx[i, j]
                actor_input[s_axis2] = yy[i, j]
                act_outp = self.actor.eval(actor_input)
                gradients = self.critic.get_io_gradient(np.concatenate((actor_input, act_outp)))[0,:]
                critic_gradient[i, j, 0] = gradients[self.s_size + u_axis1]*.5+.5
                if shown_u_size > 1:
                    critic_gradient[i, j, 1] = gradients[self.s_size + u_axis2]*.5+.5
                if shown_u_size > 2:
                    critic_gradient[i, j, 2] = gradients[self.s_size + u_axis3]*.5+.5

        ax1.imshow(critic_gradient)
        fig.tight_layout()

    def save_networks(self):
       if self.actor_save_path is not None:
           self.actor.save_weights_to(self.actor_save_path)
       if self.critic_save_path is not None:
           self.critic.save_weights_to(self.critic_save_path)
       if self.plant_save_path is not None:
           self.plant.save_weights_to(self.plant_save_path) 