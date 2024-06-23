from __future__ import annotations

import numpy as np

class Network:
    def __init__(self, p_architecture: list[int]):
        self.architecture = p_architecture
        self.weightdimensions: list[int] = []
        self.offsets: list[int] = []
        total_weights = 0
        for layer_index, layer_size in enumerate(p_architecture[1:]):
            prev_layer_size = p_architecture[layer_index]
            self.weightdimensions.append(prev_layer_size*layer_size)
            self.offsets.append(total_weights)
            total_weights += prev_layer_size*layer_size

        self.weights = np.random.uniform(-1, 1, total_weights)

    def get_weight_matrix(self, layer: int) -> np.ndarray:
        offs = self.offsets[layer]
        size = self.weightdimensions[layer]
        return self.weights[offs : offs+size].reshape(-1, self.architecture[layer])

    def forward(self, input: np.ndarray) -> np.ndarray:
        fwd_phi = lambda x: x

        N_layers = len(self.architecture)

        prev_layer = input.reshape(-1, 1)
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ prev_layer
            prev_layer = fwd_phi(out_linear)

        return prev_layer
        

    def get_gradients(self, input: np.ndarray, expected: float) -> np.ndarray:
        # fwd_phi = lambda x: np.maximum(x, 0)
        # back_phi = lambda x: np.single(x)

        fwd_phi = lambda x: x
        back_phi = lambda x: 1

        mat_input = input.reshape(-1, 1)
        N_layers = len(self.architecture)

        layers = [np.zeros(0)]*N_layers
        layers[0] = mat_input
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ layers[i]
            layers[i+1] = fwd_phi(out_linear)  # RELU
            
        output = layers[-1]

        error = output - expected
        E = .5 * np.dot(error.flatten(), error.flatten())
        print(E)

        dE_dy = error
        dE_dv = [np.zeros(0)]*N_layers
        dE_dv[-1] = dE_dy * back_phi(layers[i])
        for i in range(N_layers-2, -1, -1):
            dE_dv[i] = (dE_dv[i+1].T @ self.get_weight_matrix(i)).T * back_phi(layers[i])
            
        dE_dw = np.zeros(len(self.weights))
        for i, offs in enumerate(self.offsets):
            previous = layers[i]
            weight_size = self.weightdimensions[i]
            dE_dw[offs:offs+weight_size] = (dE_dv[i+1] @ previous.T).flatten()
        
        return dE_dw
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
    nn = Network([2, 3, 1])
    #critic_back_propagation(nn, np.array([0, 0]), 0)
    eta = 1
    for i in range(30):
        inp = np.random.uniform(-1, 1, 2)
        gradients = nn.get_gradients(inp, sum(inp))
        nn.weights -= gradients * eta
        eta *= 0.9
    #nn.train(np.array([1, 1]), 0)
    print(nn.weights)