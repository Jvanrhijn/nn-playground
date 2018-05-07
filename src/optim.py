import sys
import numpy as np
import src.layers as ly


class GDOptimizer:

    def __init__(self, learn_rate):
        self._learn_rate = learn_rate

    def optimize(self, network):
        for layer in network.layers:
            layer.weights -= layer.weight_grad * self._learn_rate
            layer.biases -= layer.biases_grad * self._learn_rate


class MomentumOptimizer:

    def __init__(self, learn_rate, mom_par, network):
        self._learn_rate = learn_rate
        self._momentum = []
        self._momentum_bias = []
        self._mom_par = mom_par
        for layer in network.layers:
            self._momentum.append(np.zeros(layer.weight_grad.shape))
            self._momentum_bias.append(np.zeros(layer.biases_grad.shape))

    def optimize(self, network):
        for idx, layer in enumerate(network.layers):
            self._momentum[idx] = self._mom_par*self._momentum[idx] - self._learn_rate*layer.weight_grad
            self._momentum_bias[idx] = self._mom_par*self._momentum_bias[idx] - self._learn_rate*layer.biases_grad
            layer.weights += self._momentum[idx]
            layer.biases += self._momentum_bias[idx]


class NAGOptimizer:

    def __init__(self, learn_rate, mom_par, network):
        self._mom_par = mom_par
        self._learn_rate = learn_rate
        self._momentum = []
        self._momentum_bias = []
        for layer in network.layers:
            self._momentum.append(np.zeros(layer.weight_grad.shape))
            self._momentum_bias.append(np.zeros(layer.biases_grad.shape))

    def optimize(self, network):
        for idx, layer in enumerate(network.layers):
            momentum_prev = self._momentum[idx]
            momentum_prev_bias = self._momentum_bias[idx]
            self._momentum[idx] = self._mom_par*self._momentum[idx] + self._learn_rate * layer.weight_grad
            self._momentum_bias[idx] = self._mom_par*self._momentum_bias[idx] + self._learn_rate * layer.biases_grad
            layer.weights -= self._mom_par * momentum_prev + (1 + self._mom_par) * self._momentum[idx]
            layer.biases -= self._mom_par * momentum_prev_bias + (1 + self._mom_par) * self._momentum_bias[idx]


class AdaGradOptimizer:

    def __init__(self, learn_rate, network, offset=10**-8):
        self._learn_rate = learn_rate
        self._offset = offset
        self._grad_square_sum = []
        self._network = network
        for layer in network.layers:
            self._grad_square_sum.append(np.zeros(layer.weights.shape))

    def optimize(self, network):
        for idx, layer in enumerate(network.layers):
            self._grad_square_sum[idx] += layer.weight_grad**2
            layer.weights -= self._learn_rate / np.sqrt(self._grad_square_sum[idx] + self._offset) * layer.weight_grad


class AdaDeltaOptimizer:

    def __init__(self, window, network, offset=10**-8):
        self._window = window
        self._network = network
        self._offset = offset
        self._grad_rms = []
        self._grad_rms_prev = []
        self._weight_rms = []
        self._weight_rms_prev = []
        self._weight_update_prev = []
        for layer in network.layers:
            self._grad_rms.append(np.zeros(layer.weights.shape))
            self._grad_rms_prev.append(np.zeros(layer.weights.shape))
            self._weight_rms.append(np.zeros(layer.weights.shape))
            self._weight_rms_prev.append(np.zeros(layer.weights.shape))
            self._weight_update_prev.append(np.zeros(layer.weights.shape))

    def optimize(self, network):
        for idx, layer in enumerate(network.layers):
            mov_av_grad = self._window * (self._grad_rms_prev[idx]**2 - self._offset) + (1 - self._window) \
                * layer.weight_grad**2
            self._grad_rms[idx] = np.sqrt(mov_av_grad + self._offset)
            mov_av_weights = self._window * (self._weight_rms_prev[idx]**2 - self._offset) + (1 - self._window) \
                * self._weight_update_prev[idx]**2
            self._weight_rms_prev[idx] = np.sqrt(mov_av_weights + self._offset)
            self._weight_update_prev[idx] = - self._weight_rms_prev[idx] / self._grad_rms[idx]
            self._grad_rms_prev[idx] = self._grad_rms[idx]
            layer.weights += self._weight_update_prev[idx]


class RMSPROptmizer:

    def __init__(self, learn_rate):
        pass

