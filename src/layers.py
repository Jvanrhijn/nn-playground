import numpy as np


class Layer:
    """Layer of a fully connected neural network"""
    def __init__(self, num_neurons, num_inputs):
        self._weights = np.random.randn(num_neurons, num_inputs)
        self._biases = np.random.randn(num_neurons)
        self._grad_inputs = 0
        self._grad_weights = 0
        self._grad_biases = 0

    def forward_pass(self, inputs):
        """Pass inputs through the network"""
        pass

    def back_propagate(self, gradients_in):
        pass

    def connect_next(self, next_layer):
        pass

    @property
    def next_layer(self):
        return self._next_layer

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @biases.setter
    def biases(self, new_biases):
        self._biases = new_biases

    @property
    def grad_weights(self):
        return self._grad_weights

    @property
    def grad_biases(self):
        return self._grad_biases

    @property
    def grad_inputs(self):
        return self._grad_inputs


class ReLuLayer(Layer):
    """Layer with ReLu activated neurons"""
    def forward_pass(self, inputs):
        outputs = np.maximum(0, np.dot(self._weights, inputs) + self._biases)
        self._grad_inputs = self._weights
        self._grad_inputs[outputs == 0] = 0
        self._grad_weights = inputs[np.newaxis, :] * np.ones(self._weights.shape)
        self._grad_weights[outputs == 0] = 0
        self._grad_biases = np.ones(self._grad_weights.shape)
        self._grad_biases[outputs == 0] = 0
        return outputs

    def back_propagate(self, gradients_in):
        pass
