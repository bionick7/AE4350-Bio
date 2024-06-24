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
                   # cast boolean to array as step-function
                   lambda f: np.asarray(np.single(f > 0)))
    
    @classmethod
    def clamped(cls, size: int) -> Layer:
        return cls(size, 
                   lambda x: np.minimum(np.maximum(x, -1), 1), 
                   # cast boolean to array as step-function
                   lambda f: np.asarray(np.single(np.abs(f) < 1)))
    
    @classmethod
    def sigmoid(cls, size: int) -> Layer:
        return cls(size, 
                   lambda x: 1 / (1 + np.exp(-x)), 
                   lambda f: f * (1 - f))
    
    @classmethod
    def tanh(cls, size: int) -> Layer:
        return cls(size, 
                   lambda x: np.tanh(x), 
                   lambda f: 1-f*f)

class Network:
    def __init__(self, p_architecture: list[Layer], 
                 w_clamp: tuple[float, float]=(-1,1), b_clamp: tuple[float, float]=(-1,1)):
        self.w_min, self.w_max = w_clamp
        self.b_min, self.b_max = b_clamp
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
            weights_in_layer = (prev_layer_size+1)*layer.size
            self.weightdimensions.append(weights_in_layer)
            self.offsets.append(total_weights)
            total_weights += weights_in_layer

        self.weights = np.zeros(total_weights)
        self._initialize()

    def _initialize(self) -> None:
        for i in range(len(self.architecture)-1):
            offs = self.offsets[i]
            size = self.weightdimensions[i]
            layer_size = self.architecture[i].size + 1
            biases = size // layer_size
            matrix = np.hstack((np.random.uniform(self.w_min, self.w_max, size - biases), 
                                np.random.uniform(self.b_min, self.b_max, size // layer_size)))
            self.weights[offs:offs+size] = matrix.flatten().copy()

    def update_weights(self, delta: np.ndarray) -> None:
        self.weights += delta
        for i in range(len(self.architecture)-1):
            offs = self.offsets[i]
            size = self.weightdimensions[i]
            layer_size = self.architecture[i].size + 1
            matrix = self.weights[offs:offs+size].reshape(-1, layer_size)
            matrix[:, -1] = np.maximum(np.minimum(matrix[:, -1], self.b_max), self.b_min)
            matrix[:,:-1] = np.maximum(np.minimum(matrix[:,:-1], self.w_max), self.w_min)
            self.weights[offs:offs+size] = matrix.flatten()

    def get_weight_matrix(self, layer: int) -> np.ndarray:
        offs = self.offsets[layer]
        size = self.weightdimensions[layer]
        return self.weights[offs : offs+size].reshape(-1, self.architecture[layer].size+1)

    def eval(self, input: np.ndarray) -> np.ndarray:
        N_layers = len(self.architecture)

        prev_layer = np.append(input, 1).reshape(-1, 1)
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ prev_layer
            prev_layer = np.append(self.architecture[i+1].fwd(out_linear), [[1]], axis=0)

        return prev_layer[:-1]

    def get_gradients(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mat_input = input.reshape(-1, 1)
        N_layers = len(self.architecture)

        layers = [np.zeros(0)]*N_layers
        layers[0] = np.append(mat_input, [[1]], axis=0)
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ layers[i]
            layers[i+1] = np.append(self.architecture[i+1].fwd(out_linear), [[1]], axis=0)
            
        output = layers[-1][:-1]

        # Gradient w.r.t layers
        dy_dv = [np.zeros(0)]*N_layers
        dy_dv[-1] = np.diag(self.architecture[-1].back(output).flatten())
        for i in range(N_layers-2, -1, -1):
            activation_derivative = self.architecture[i].back(layers[i][:-1]).T
            dy_dv[i] = (dy_dv[i+1] @ self.get_weight_matrix(i))[:,:-1] * activation_derivative
            
        # Gradient w.r.t weights
        dy_dw = np.zeros((len(output), len(self.weights)))
        for i, offs in enumerate(self.offsets):
            previous = layers[i]
            weight_size = self.weightdimensions[i]
            for j in range(len(output)):
                dy_dw[j,offs:offs+weight_size] = (previous @ dy_dv[i+1][j:j+1]).T.flatten()
        
        return output, dy_dw


def check_gradients(nn: Network, show:bool=False) -> bool:

    input_dim = nn.architecture[0].size
    inp = np.random.uniform(-7, 7, (input_dim,))
    output, J = nn.get_gradients(inp)
    EPSILON = 1e-2
    for output_index in range(len(J)):
        for weight_index in range(len(J[output_index])):
            w = nn.weights[weight_index]
            nn.weights[weight_index] = w + EPSILON
            y_plus = nn.eval(inp)[output_index,0]
            nn.weights[weight_index] = w - EPSILON
            y_minus = nn.eval(inp)[output_index,0]
            nn.weights[weight_index] = w
            finite_diff = (y_plus - y_minus) / (2*EPSILON)
            if show:
                print(finite_diff, J[output_index,weight_index])
            discrepency = abs(finite_diff - J[output_index,weight_index])
            if discrepency > 1e-3:
                print(f"Discrepency ({discrepency}) to finite differencees found:")
                return False

    return True


def sin_test(nn: Network) -> None:
    import matplotlib.pyplot as plt

    epochs = 100
    epoch_size = 20

    learning_xx = np.linspace(-7, 7, epoch_size)
    learning_progress = np.zeros(epochs)

    prev_epoch_error = 1e4

    for epoch in range(epochs):
        eta = 2
        avg_error = 0
        cum_delta_weights = np.zeros(len(nn.weights))
        for index in range(epoch_size):
            #inp = np.random.uniform(-7, 7, 1)
            inp = learning_xx[index]
            inp = np.random.uniform(-7, 7, (1,))
            expected = np.sin(inp)
            output, J = nn.get_gradients(inp)
            e = output - expected.reshape(-1, 1)
            mu = min(0.99, max(0.01, 1 - epoch / 20))
            delta_weights = -np.linalg.solve(J.T@J - mu * np.eye(J.shape[1]), J.T@e).flatten()
            nn.update_weights(delta_weights)
            cum_delta_weights += delta_weights
            E = .5 * (e.T@e)[0,0]
            avg_error += E/epoch_size

        #nn.weights += cum_delta_weights
        
        avg_error_2 = 0
        for index in range(epoch_size):
            #inp = np.random.uniform(-7, 7, 1)
            inp = learning_xx[index]
            expected = np.sin(inp)
            e = nn.eval(inp) - expected.reshape(-1, 1)
            E = .5 * (e.T@e)[0,0]
            avg_error_2 += E/epoch_size
        
        if avg_error_2 > avg_error:  # Discard if the error increases
            #nn.weights -= cum_delta_weights
            eta *= 0.95
        else:
            prev_epoch_error = avg_error
            eta /= 0.95
        learning_progress[epoch] = min(avg_error, prev_epoch_error)
        #print(min(E2, E))
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    sample_xx = np.linspace(-7, 7, 100)
    sample_yy = np.zeros(100)
    for i in range(100):
        sample_yy[i] = nn.eval(sample_xx[i])[0,0]
    ax1.plot(sample_xx, sample_yy)
    ax1.plot(sample_xx, np.sin(sample_xx))
    ax1.scatter(learning_xx, np.sin(learning_xx))

    ax2 = fig1.add_subplot(212)
    ax2.plot(learning_progress)
    ax2.set_ylim((0, 1))

    plt.show()

    #for i in range(10):
    #    inp = learning_xx[index]
    #    expected = np.sin(inp)
    #    out = nn.forward(inp)
    #    e = nn.forward(inp) - expected.reshape(-1, 1)
    #    E = .5 * (e.T@e)[0,0]
    #    print(E)

if __name__ == "__main__":
    nn = Network([
        Layer.linear(1),
        Layer.tanh(10),
        Layer.linear(1)
    ], (-1, 1), (-7, 7))

    check_gradients(nn)
    sin_test(nn)

        
