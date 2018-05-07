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
        self._mom_par = mom_par
        for layer in network.layers:
            self._momentum.append(np.zeros(layer.weight_grad.shape))

    def optimize(self, network):
        for idx, layer in enumerate(network.layers):
            self._momentum[idx] = self._mom_par*self._momentum[idx] - self._learn_rate*layer.weight_grad
            layer.weights += self._momentum[idx]


class NAGOptimizer:

    def __init__(self, learn_rate, mom_par, network):
        self._mom_par = mom_par
        self._learn_rate = learn_rate
        self._momentum = []
        for layer in network._layers:
            self._momentum.append(np.zeros(layer.weight_grad.shape))

    def optimize(self, network):
        for idx, layer in enumerate(network.layers):
            momentum_prev = self._momentum[idx]
            self._momentum[idx] = self._mom_par*self._momentum[idx] + self._learn_rate * layer.weight_grad
            layer.weights -= self._mom_par * momentum_prev + (1 + self._mom_par) * self._momentum[idx]


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


class RMSPROptmizer:

    def __init__(self, learn_rate):
        pass

