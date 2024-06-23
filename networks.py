from __future__ import annotations
from typing import Callable

import numpy as np



class Layer:
    def __init__(self, p_size: int, p_fwd: Callable[[np.ndarray], np.ndarray], 
                 p_back: Callable[[np.ndarray], np.ndarray]):
        self.size = p_size
        self.fwd = p_fwd
        self.back = p_back  # Derivative gets calculated off of f(x)
        
    @classmethod
    def linear(cls, size: int) -> Layer:
        return cls(size, 
                   lambda x: x, 
                   lambda f: np.ones(f.shape))

    @classmethod
    def relu(cls, size: int) -> Layer:
        return cls(size, 
                   lambda x: np.maximum(x, 0), 
                   lambda f: np.asarray(np.single(f > 0)))  # cast boolean to array as step-function
    @classmethod
    def sigmoid(cls, size: int) -> Layer:
        return cls(size, 
                   lambda x: 1 / np.exp(-x), 
                   lambda f: f * (1 - f))

class Network:
    def __init__(self, p_architecture: list[Layer]):
        self.architecture = p_architecture
        self.weightdimensions: list[int] = []
        self.offsets: list[int] = []
        self.activation_functions = []
        self.activation_backprops = []

        total_weights = 0
        for layer_index, layer in enumerate(self.architecture[1:]):
            self.activation_functions.append(layer.fwd)
            self.activation_backprops.append(layer.back)
            prev_layer_size = self.architecture[layer_index].size
            self.weightdimensions.append(prev_layer_size*layer.size)
            self.offsets.append(total_weights)
            total_weights += prev_layer_size*layer.size

        self.weights = np.random.uniform(-1, 1, total_weights)

    def get_weight_matrix(self, layer: int) -> np.ndarray:
        offs = self.offsets[layer]
        size = self.weightdimensions[layer]
        return self.weights[offs : offs+size].reshape(-1, self.architecture[layer].size)

    def forward(self, input: np.ndarray) -> np.ndarray:
        fwd_phi = lambda x: x

        N_layers = len(self.architecture)

        prev_layer = input.reshape(-1, 1)
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ prev_layer
            prev_layer = fwd_phi(out_linear)

        return prev_layer

    def get_gradients(self, input: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mat_input = input.reshape(-1, 1)
        N_layers = len(self.architecture)

        layers = [np.zeros(0)]*N_layers
        layers[0] = mat_input
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ layers[i]
            layers[i+1] = self.architecture[i+1].fwd(out_linear)  # RELU
            
        output = layers[-1]
        error = output - expected.reshape(-1, 1)

        # Gradient w.r.t layers
        dy_dv = [np.zeros(0)]*N_layers
        dy_dv[-1] = np.diag(self.architecture[-1].back(output).flatten())
        for i in range(N_layers-2, -1, -1):
            dy_dv[i] = dy_dv[i+1] @ self.get_weight_matrix(i) * self.architecture[i].back(layers[i]).T
            
        # Gradient w.r.t weights
        dy_dw = np.zeros((len(output), len(self.weights)))
        for i, offs in enumerate(self.offsets):
            previous = layers[i]
            weight_size = self.weightdimensions[i]
            for j in range(len(output)):
                dy_dw[j,offs:offs+weight_size] = (previous @ dy_dv[i+1][j:j+1]).T.flatten()
        
        return error, dy_dw
        #error2 = clone.forward(input) - expected
        #E2 = .5 * np.dot(error2.flatten(), error2.flatten())
        #if E2 < E:
        #    self.copy(clone)
        #    return E2
        #else:
        #    clone.copy(self)
        #    return E
            

def critic_fwd_propagation(net: Network, input: np.ndarray) -> float:
    hidden_linear = net.weights[0] @ input
    hidden = np.maximum(hidden_linear, 0)  # RELU
    output_linear = net.weights[1] @ hidden
    output = np.maximum(output_linear, 0)  # RELU
    return output[0]


def critic_back_propagation(net: Network, input: np.ndarray, expected: float):
    input = input.reshape(-1, 1)
    hidden_linear = net.weights[0] @ input
    hidden = np.maximum(hidden_linear, 0)  # RELU
    output_linear = net.weights[1] @ hidden
    output = np.maximum(output_linear, 0)  # RELU

    error = output - expected
    E = .5 * (error)**2

    dE_dy = (error - 1)
    dE_dv2 = dE_dy * np.single(output > 0)
    dE_dv1 = (dE_dv2 @ net.weights[1]).T * np.single(hidden > 0)
    dE_dw_2 = dE_dv2 @ hidden.T
    dE_dw_1 = dE_dv1 @ input.T

    net.weights[0] += dE_dw_1 * E * 0.1
    net.weights[1] += dE_dw_2 * E * 0.1


if __name__ == "__main__":
    nn = Network([
        Layer.linear(2),
        Layer.linear(3),
        Layer.linear(2),
    ])
    #critic_back_propagation(nn, np.array([0, 0]), 0)
    eta = 2
    for i in range(10):
        inp = np.random.uniform(-1, 1, 2)
        e, J = nn.get_gradients(inp, np.array([sum(inp), inp[0]-inp[1]]))
        E = .5 * (e.T@e)
        nn.weights -= np.linalg.solve(J.T@J - 0.1 * np.eye(J.shape[1]), J.T@e).flatten()
        eta *= 0.9
    #nn.train(np.array([1, 1]), 0)
    print(nn.weights)