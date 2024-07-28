from __future__ import annotations
from typing import Callable
from math import prod
from copy import copy, deepcopy

import os.path
import numpy as np

def levenberg_marquardt(grad: np.ndarray, e: np.ndarray, mu: float):
    return -np.linalg.solve(grad.T@grad - mu * np.eye(grad.shape[1]), grad.T@e).flatten()


class Network:
    weights: np.ndarray

    def get_layer_size(self, layer) -> int:
        raise NotImplementedError("Not available in base class")

    def update_weights(self, delta: np.ndarray) -> None:
        raise NotImplementedError("Not available in base class")

    def eval(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not available in base class")
        
    def get_weight_gradient(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Not available in base class")
    
    def get_io_gradient(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not available in base class")

    def load_weights_from(self, filepath: str):
        self.weights *= 0
        self.update_weights(np.loadtxt(os.path.join("saved_networks", filepath)))
    
    def save_weights_to(self, filepath: str):
        np.savetxt(os.path.join("saved_networks", filepath), self.weights)

    def learn_gd(self, beta: float, input: np.ndarray|float, expected_output: np.ndarray|float
                 ) -> tuple[np.ndarray, float]:
        if isinstance(input, float):
            input = np.array([input])
        if isinstance(expected_output, float):
            expected_output = np.array([expected_output])
        outp, grad = self.get_weight_gradient(input)
        e = outp - expected_output
        weight_delta = (grad.T @ e).flatten() * -beta
        self.update_weights(weight_delta)

        return outp, 0.5 * np.dot(e, e)
    
    def learn_ml(self, mu: float, input: np.ndarray|float, expected_output: np.ndarray|float
                 ) -> tuple[np.ndarray, float]:
        if isinstance(input, float):
            input = np.array([input])
        if isinstance(expected_output, float):
            expected_output = np.array([expected_output])
        outp, grad = self.get_weight_gradient(input)
        e = outp - expected_output
        weight_delta = levenberg_marquardt(grad, e, mu)
        self.update_weights(weight_delta)

        return outp, 0.5 * (e.T@e).flatten()[0]

class RBFNN(Network):
    def __init__(self, p_inpsize: int, p_size: int, p_outsize: int,
                 w_clamp: tuple[float, float]=(-1,1), b_clamp: tuple[float, float]=(-1,1)):
        self.inpsize = p_inpsize
        self.size = p_size
        self.outsize = p_outsize
        self.w_min, self.w_max = w_clamp
        self.b_min, self.b_max = b_clamp
        self.centers = np.zeros((self.size, self.inpsize))
        self.inv_stdevs = np.ones((self.size, self.inpsize))
        self.weights = np.random.uniform(self.w_min, self.w_max, (self.size + 1) * self.outsize)

    def get_layer_size(self, layer) -> int:
        return (self.inpsize, self.size, self.outsize)[layer]
    
    def set_rbfs(self, p_centers: np.ndarray, p_stddevs: np.ndarray|None=None) -> None:
        if p_stddevs is None:
            self.inv_stdevs = np.ones((self.size, self.inpsize))
        else:
            self.inv_stdevs = 1/p_stddevs.reshape(self.size, self.inpsize)
        self.centers = p_centers
        
    def update_weights(self, delta: np.ndarray) -> None:
        self.weights += delta
        matrix = self.weights.reshape(self.outsize, -1)
        matrix[:, -1] = np.maximum(np.minimum(matrix[:, -1], self.b_max), self.b_min)
        matrix[:,:-1] = np.maximum(np.minimum(matrix[:,:-1], self.w_max), self.w_min)
        self.weights = matrix.flatten()

    def eval(self, input: np.ndarray) -> np.ndarray:
        dist = input[np.newaxis,:] - self.centers
        rbf_out = np.exp(-np.sum(np.square(dist*self.inv_stdevs), axis=1))
        out = self.weights.reshape(self.outsize, -1) @ np.append(rbf_out, 1)
        return out
    
    def get_weight_gradient(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist = input[np.newaxis,:] - self.centers
        rbf_out = np.exp(-np.sum(np.square(dist*self.inv_stdevs), axis=1))
        output = self.weights.reshape(self.outsize, -1) @ np.append(rbf_out, 1)
        
        # Gradient w.r.t layers
        dy_dout = np.eye(self.outsize)
            
        # Gradient w.r.t weights
        dy_dw = np.zeros((self.outsize, len(self.weights)))
        for j in range(self.outsize):
            previous_layer = np.append(rbf_out[:,np.newaxis], 1)[:,np.newaxis]
            dy_dw[j] = (previous_layer @ dy_dout[j:j+1]).T.flatten()
        
        return output, dy_dw

    def get_io_gradient(self, input: np.ndarray) -> np.ndarray:
        dist = input[np.newaxis,:] - self.centers
        rbf_out = np.exp(-np.sum(np.square(dist*self.inv_stdevs), axis=1))
        
        # Gradients
        dy_dv = self.weights.reshape(self.outsize, -1)[:,:-1]
        dv_din = -2 * dist * self.inv_stdevs*self.inv_stdevs * rbf_out[:,np.newaxis]
        return dy_dv @ dv_din

    @classmethod
    def grid_spaced(cls, output_size: int, *spacings: np.ndarray,
                    w_clamp: tuple[float, float]=(-1,1), b_clamp: tuple[float, float]=(-1,1)) -> RBFNN:
        ''' Assumes equal spacing '''
        input_size = len(spacings)
        rbf_size = prod([len(spacing) for spacing in spacings])
        res = RBFNN(input_size, rbf_size, output_size, w_clamp, b_clamp)
        rbf_coords = np.meshgrid(*spacings)
        centers = np.vstack([x.flatten() for x in rbf_coords]).T
        stdevs = np.zeros(input_size)
        for i, spacing in enumerate(spacings):
            stdevs[i] = (np.max(spacing) - np.min(spacing)) / len(spacing)
        res.set_rbfs(centers, np.ones(centers.shape) * stdevs*1.5)
        return res

class Layer:
    def __init__(self, p_size: int, p_id: str,
                 p_fwd: Callable[[np.ndarray], np.ndarray], 
                 p_back: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.id = p_id
        self.size = p_size
        self.fwd = p_fwd
        self.back = p_back  # Derivative gets calculated off of f(x)
        
    @classmethod
    def gaussian(cls, size: int) -> Layer:
        return cls(size, "gaussian",
                   lambda x: np.exp(-x*x), 
                   lambda x, f: -2*x*f)

    @classmethod
    def linear(cls, size: int) -> Layer:
        return cls(size, "linear",
                   lambda x: x, 
                   lambda x, f: np.ones(f.shape))

    @classmethod
    def relu(cls, size: int) -> Layer:
        return cls(size, "relu",
                   lambda x: np.maximum(x, 0), 
                   # cast boolean to array as step-function
                   lambda x, f: np.asarray(np.single(f > 0)))
    
    @classmethod
    def clamped(cls, size: int) -> Layer:
        return cls(size, "clamped",
                   lambda x: np.minimum(np.maximum(x, -1), 1), 
                   # cast boolean to array as step-function
                   lambda x, f: np.asarray(np.single(np.abs(f) < 1)))
    
    @classmethod
    def sigmoid(cls, size: int) -> Layer:
        return cls(size, "sigmoid",
                   lambda x: 1 / (1 + np.exp(-x)), 
                   lambda x, f: f * (1 - f))
    
    @classmethod
    def tanh(cls, size: int) -> Layer:
        return cls(size, "tanh",
                   lambda x: np.tanh(x), 
                   lambda x, f: 1-f*f)
    
    @classmethod
    def sin(cls, size: int, omega:float=np.pi) -> Layer:
        return cls(size, "tanh",
                   lambda x: np.sin(x*omega), 
                   lambda x, f: np.cos(x*omega)*omega)
    

    def __copy__(self) -> Layer:
        return Layer(self.size, self.id, self.fwd, self.back)

    def __repr__(self) -> str:
        return f"[Layer {self.id} x {self.size}]"

class FFNN(Network):
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

    def get_layer_size(self, layer) -> int:
        return self.architecture[layer].size
    
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

    def set_weight_matrix(self, layer: int, weights: np.ndarray) -> None:
        offs = self.offsets[layer]
        size = self.weightdimensions[layer]
        self.weights[offs : offs+size] = weights.flatten()

    def eval(self, input: np.ndarray) -> np.ndarray:
        assert len(input) == self.architecture[0].size
        N_layers = len(self.architecture)

        prev_layer = np.append(input, 1).reshape(-1, 1)
        for i in range(N_layers-1):
            out_linear = self.get_weight_matrix(i) @ prev_layer
            prev_layer = np.append(self.architecture[i+1].fwd(out_linear), [[1]], axis=0)

        return prev_layer[:-1].flatten()

    def get_weight_gradient(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(input) == self.architecture[0].size
        mat_input = input.reshape(-1, 1)
        N_layers = len(self.architecture)

        layers = [np.zeros(0)]*N_layers
        layers_lin = [np.zeros(0)]*N_layers
        layers[0] = np.append(mat_input, [[1]], axis=0)
        for i in range(N_layers-1):
            layers_lin[i+1] = self.get_weight_matrix(i) @ layers[i]
            layers[i+1] = np.append(self.architecture[i+1].fwd(layers_lin[i+1]), [[1]], axis=0)
            
        output = layers[-1][:-1]

        # Gradient w.r.t layers
        dy_dv = [np.zeros(0)]*N_layers
        dy_dv[-1] = np.diag(self.architecture[-1].back(layers_lin[-1], output).flatten())
        for i in range(N_layers-2, -1, -1):
            activation_derivative = self.architecture[i].back(layers_lin[i], layers[i][:-1]).T
            dy_dv[i] = (dy_dv[i+1] @ self.get_weight_matrix(i))[:,:-1] * activation_derivative
            
        # Gradient w.r.t weights
        dy_dw = np.zeros((len(output), len(self.weights)))
        for i, offs in enumerate(self.offsets):
            previous = layers[i]
            weight_size = self.weightdimensions[i]
            for j in range(len(output)):
                dy_dw[j,offs:offs+weight_size] = (previous @ dy_dv[i+1][j:j+1]).T.flatten()
        
        return output.flatten(), dy_dw

    def get_io_gradient(self, input: np.ndarray) -> np.ndarray:
        assert len(input) == self.architecture[0].size
        mat_input = input.reshape(-1, 1)
        N_layers = len(self.architecture)

        layers = [np.zeros(0)]*N_layers
        layers_lin = [np.zeros(0)]*N_layers
        layers[0] = np.append(mat_input, [[1]], axis=0)
        for i in range(N_layers-1):
            layers_lin[i+1] = self.get_weight_matrix(i) @ layers[i]
            layers[i+1] = np.append(self.architecture[i+1].fwd(layers_lin[i+1]), [[1]], axis=0)
            
        output = layers[-1][:-1]

        # Gradient w.r.t layers
        dy_dv = [np.zeros(0)]*N_layers
        dy_dv[-1] = np.diag(self.architecture[-1].back(layers_lin[-1], output).flatten())
        for i in range(N_layers-2, -1, -1):
            activation_derivative = self.architecture[i].back(layers_lin[i], layers[i][:-1]).T
            dy_dv[i] = (dy_dv[i+1] @ self.get_weight_matrix(i))[:,:-1] * activation_derivative

        return dy_dv[0]
    
    def show_weights(self):
        import matplotlib.pyplot as plt

        N_layers = len(self.architecture) - 1
        fig = plt.figure()
        axes = fig.subplots(N_layers, 1, sharex=True)
        if N_layers > 1:
            for i in range(N_layers):
                c = axes[i].imshow(self.get_weight_matrix(i))
        else:
            c = axes.imshow(self.get_weight_matrix(0))
        fig.colorbar(c)
        

    @classmethod
    def stack(cls, *args: FFNN) -> FFNN:
        assert len(args) > 0
        architecture: list[Layer] = deepcopy(args[0].architecture)
        w_min, w_max, b_min, b_max = args[0].w_min, args[0].w_max, args[0].b_min, args[0].b_max
        for nn in args[1:]:
            assert len(nn.architecture) == len(architecture)
            assert all([x.id == y.id for x, y in zip(architecture, nn.architecture)])
            for i in range(len(architecture)):
                architecture[i].size += nn.architecture[i].size
            if nn.w_min < w_min: w_min = nn.w_min
            if nn.w_max > w_max: w_max = nn.w_max
            if nn.b_min < b_min: b_min = nn.b_min
            if nn.b_max > b_max: b_max = nn.b_max
        
        res = FFNN(architecture, (w_min, w_max), (b_min, b_max))
        
        for i in range(len(res.architecture)-1):
            offs = res.offsets[i]
            size = res.weightdimensions[i]
            layer_size = res.architecture[i].size + 1
            matrix = res.weights[offs:offs+size].reshape(-1, layer_size) * 0
            
            inp_offset = 0
            outp_offset = 0
            for nn in args:
                nn_weights = nn.get_weight_matrix(i)
                next_inp_offset = inp_offset + nn_weights.shape[0]
                next_outp_offset = outp_offset + nn_weights.shape[1] - 1  # excluding bias
                matrix[inp_offset:next_inp_offset,outp_offset:next_outp_offset] += nn_weights[:,:-1]
                matrix[inp_offset:next_inp_offset, -1] += nn_weights[:, -1]
                inp_offset = next_inp_offset
                outp_offset = next_outp_offset
            
            res.weights[offs:offs+size] = matrix.flatten()
        
        return res
    
    def __repr__(self) -> str:
        return " -> ".join([str(x) for x in self.architecture])

def check_weight_gradients(nn: Network, show:bool=False) -> bool:
    input_dim = nn.get_layer_size(0)
    inp = np.random.uniform(-1, 1, (input_dim,))
    output, J = nn.get_weight_gradient(inp)
    EPSILON = 1e-2
    for output_index in range(len(J)):
        for weight_index in range(len(J[output_index])):
            w = nn.weights[weight_index]
            nn.weights[weight_index] = w + EPSILON
            y_plus = nn.eval(inp)[output_index]
            nn.weights[weight_index] = w - EPSILON
            y_minus = nn.eval(inp)[output_index]
            nn.weights[weight_index] = w
            finite_diff = (y_plus - y_minus) / (2*EPSILON)
            if show:
                print(finite_diff, J[output_index,weight_index])
            discrepency = abs(finite_diff - J[output_index,weight_index])
            if discrepency > 1e-3:
                print(f"Discrepency ({discrepency}) to finite differencees (weights) found:")
                return False

    return True


def check_io_gradients(nn: Network, show:bool=False) -> bool:
    input_dim = nn.get_layer_size(0)
    inp = np.random.uniform(-1, 1, input_dim)
    output = nn.eval(inp)
    grad = nn.get_io_gradient(inp)
    EPSILON = 1e-2
    for output_index in range(len(output)):
        for input_index in range(input_dim):
            new_inp = inp.copy()
            new_inp[input_index] += EPSILON
            y_plus = nn.eval(new_inp)
            new_inp[input_index] -= EPSILON*2
            y_minus = nn.eval(new_inp)
            finite_diff = (y_plus - y_minus) / (2*EPSILON)
            if show:
                print(finite_diff, grad[output_index, input_index])
            discrepency = abs(finite_diff - grad[output_index, input_index])
            if discrepency > 3e-3:
                print(f"Discrepency ({discrepency}) to finite differencees (IO) found:")
                return False
    print("IO gradient checks out")
    return True


def sin_test(nn: Network) -> None:
    import matplotlib.pyplot as plt

    epochs = 20
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
            #inp = learning_xx[index]
            inp = np.random.uniform(-7, 7, (1,))
            expected = np.sin(inp)
            output, J = nn.get_weight_gradient(inp)
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
            inp = learning_xx[index,np.newaxis]
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
    sample_xx = np.linspace(-7, 7, 1000)
    sample_yy = np.zeros(1000)
    for i in range(1000):
        sample_yy[i] = nn.eval(sample_xx[i,np.newaxis])[0]
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
    ffnn = FFNN([
        Layer.linear(1),
        Layer.tanh(20),
        Layer.linear(1)
    ], (-1, 1), (-7, 7))

    check_weight_gradients(ffnn)
    check_io_gradients(ffnn)
    sin_test(ffnn)

    N_rbf = 15
    rbfnn = RBFNN(1, N_rbf, 1, (-1, 1), (0, 0))
    rbfnn.set_rbfs(np.linspace(-7, 7, N_rbf)[:,np.newaxis], np.ones(N_rbf) * 1)
    
    #sin_test(rbfnn)
    #check_weight_gradients(rbfnn)
    #check_io_gradients(rbfnn)
