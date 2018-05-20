import sys
import numpy as np
import src.layers as ly


class Optimizer:
    def __init__(self, network):
        self._network = network

    def optimize(self):
        pass


class GDOptimizer(Optimizer):

    def __init__(self, learn_rate, network):
        super().__init__(network)
        self._learn_rate = learn_rate

    def optimize(self):
        for layer in self._network.layers:
            layer.weights -= layer.weight_grad * self._learn_rate


class MomentumOptimizer(Optimizer):

    def __init__(self, learn_rate, mom_par, network):
        super().__init__(network)
        self._learn_rate = learn_rate
        self._momentum = []
        self._mom_par = mom_par
        for layer in self._network.layers:
            self._momentum.append(np.zeros(layer.weight_grad.shape))

    def optimize(self):
        for idx, layer in enumerate(self._network.layers):
            self._momentum[idx] = self._mom_par*self._momentum[idx] - self._learn_rate*layer.weight_grad
            layer.weights += self._momentum[idx]


class NAGOptimizer(Optimizer):

    def __init__(self, learn_rate, mom_par, network):
        super().__init__(network)
        self._mom_par = mom_par
        self._learn_rate = learn_rate
        self._momentum = []
        for layer in self._network.layers:
            self._momentum.append(np.zeros(layer.weight_grad.shape))

    def optimize(self):
        for idx, layer in enumerate(self._network.layers):
            momentum_prev = self._momentum[idx]
            self._momentum[idx] = self._mom_par*self._momentum[idx] + self._learn_rate * layer.weight_grad
            layer.weights -= self._mom_par * momentum_prev + (1 + self._mom_par) * self._momentum[idx]


class AdaGradOptimizer(Optimizer):

    def __init__(self, learn_rate, network, offset=10**-8):
        super().__init__(network)
        self._learn_rate = learn_rate
        self._offset = offset
        self._grad_square_sum = []
        self._network = network
        for layer in network.layers:
            self._grad_square_sum.append(np.zeros(layer.weights.shape))

    def optimize(self):
        for idx, layer in enumerate(self._network.layers):
            self._grad_square_sum[idx] += layer.weight_grad**2
            layer.weights -= self._learn_rate / np.sqrt(self._grad_square_sum[idx] + self._offset) * layer.weight_grad


class AdaDeltaOptimizer(Optimizer):

    def __init__(self, window, network, offset=10**-8):
        self._window = window
        self._network = network
        self._offset = offset
        self._grad_weights_mov_av = [np.zeros(layer.weight_grad.shape) for layer in network.layers]
        self._weight_step_mov_av = [np.zeros(layer.weight_grad.shape) for layer in network.layers]
        self._weight_step_prev = [np.zeros(layer.weights.shape) for layer in network.layers]

    def optimize(self):
        for idx, layer in enumerate(self._network.layers):
            self._grad_weights_mov_av[idx] = self._update_mov_av(self._grad_weights_mov_av[idx], layer.weight_grad)
            grad_weights_rms = self._compute_rms(self._grad_weights_mov_av[idx])
            self._weight_step_mov_av[idx] = self._update_mov_av(self._weight_step_mov_av[idx],
                                                                self._weight_step_prev[idx])
            weight_step_rms = self._compute_rms(self._weight_step_mov_av[idx])
            self._weight_step_prev[idx] = self._get_step(weight_step_rms, grad_weights_rms, layer.weight_grad)
            layer.weights += self._weight_step_prev[idx]

    def _update_mov_av(self, mov_av, new):
        return self._window * mov_av + (1 - self._window) * new**2

    def _compute_rms(self, mov_av):
        return np.sqrt(mov_av + self._offset)

    @staticmethod
    def _get_step(step_rms, grad_rms, grad):
        return -step_rms / grad_rms * grad


class RMSpropOptmizer(Optimizer):

    def __init__(self, learn_rate, window, network, offset=10**-8):
        super().__init__(network)
        self._learn_rate = learn_rate
        self._window = window
        self._offset = offset
        self._network = network
        self._grad_weights_rms = [np.zeros(layer.weight_grad.shape) for layer in network.layers]

    def optimize(self):
        for idx, layer in enumerate(self._network.layers):
            self._grad_weights_rms[idx] = self._update_mov_av(self._grad_weights_rms[idx], layer.weight_grad**2)
            layer.weights -= self._learn_rate / np.sqrt(self._grad_weights_rms[idx] + self._offset) * layer.weight_grad

    def _update_mov_av(self, mov_av, grads):
        return self._window * mov_av + (1 - self._window) * grads


class AdamOptimizer(Optimizer):

    def __init__(self, learn_rate, window, window_sq, network, offset=10**-8):
        super().__init__(network)
        self._time_step = 0
        self._learn_rate = learn_rate
        self._window = window
        self._window_sq = window_sq
        self._network = network
        self._offset = offset
        self._mov_av_grad_weight = [np.zeros(layer.weight_grad.shape) for layer in network.layers]
        self._mov_av_grad_weight_sq = [np.zeros(layer.weight_grad.shape) for layer in network.layers]

    def optimize(self):
        self._time_step += 1
        for idx, layer in enumerate(self._network.layers):
            self._mov_av_grad_weight[idx] = self._update_mov_av(self._mov_av_grad_weight[idx], layer.weight_grad,
                                                                self._window)
            self._mov_av_grad_weight_sq[idx] = self._update_mov_av_sq(self._mov_av_grad_weight_sq[idx],
                                                                      layer.weight_grad, self._window_sq)
            mov_av_grad_weight_corr = self._mov_av_grad_weight[idx] / (1 - self._window**self._time_step)
            mov_av_grad_weight_sq_corr = self._mov_av_grad_weight_sq[idx] / (1 - self._window_sq**self._time_step)
            layer.weights -= self._learn_rate / (np.sqrt(mov_av_grad_weight_sq_corr) + self._offset) \
                             * mov_av_grad_weight_corr

    @staticmethod
    def _update_mov_av_sq(rms, grads, window):
        return window * rms + (1 - window) * grads**2

    @staticmethod
    def _update_mov_av(current, grads, window):
        return window * current + (1 - window) * grads


class NadamOptimizer(AdamOptimizer):

    def __init__(self, learn_rate, window, window_sq, network, offset=10**-8):
        super().__init__(learn_rate, window, window_sq, network, offset=offset)

    def optimize(self):
        self._time_step += 1
        for idx, layer in enumerate(self._network.layers):
            self._mov_av_grad_weight[idx] = self._update_mov_av(self._mov_av_grad_weight[idx], layer.weight_grad,
                                                                self._window)
            self._mov_av_grad_weight_sq[idx] = self._update_mov_av_sq(self._mov_av_grad_weight_sq[idx],
                                                                      layer.weight_grad, self._window_sq)

            mov_av_grad_weight_corr = self._mov_av_grad_weight[idx] / (1 - self._window**self._time_step)
            mov_av_grad_weight_sq_corr = self._mov_av_grad_weight_sq[idx] / (1 - self._window_sq**self._time_step)
            step = -self._learn_rate / (np.sqrt(mov_av_grad_weight_sq_corr) + self._offset) \
                   * (self._window*mov_av_grad_weight_corr +
                      (1 - self._window)*layer.weight_grad/(1 - self._window**self._time_step))
            layer.weights += step

