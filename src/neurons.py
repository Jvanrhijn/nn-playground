import numpy as np
import src.activators as act


class Neuron:
    def __init__(self, activation, num_weights):
        self._activation = activation.activator
        self._gradient = activation.gradient
        self._weights = np.zeros([num_weights])
        self._inputs = 0
        self._output = 0

    @property
    def activation(self):
        return self._activation

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        assert new_weights.shape == self._weights.shape
        self._weights = new_weights

    def init_random(self):
        self._weights = np.random.randn(len(self._weights))*0.01
        return self

    def activate(self, inp):
        self._inputs = inp
        return self._activation(np.dot(self._weights, inp))

    def back_prop(self, grad_in):
        """Back propagate through the neuron, returning the
        gradient wrt to the inputs and wrt to the weights"""
        gradient_activation = self._gradient(np.dot(self._inputs, self._weights))
        grad_inputs = gradient_activation * self._weights * grad_in
        grad_weights = gradient_activation * self._inputs * grad_in
        return grad_inputs, grad_weights
