import numpy as np


class Layer:
    """Layer of a fully connected neural network"""
    def __init__(self, num_neurons, num_inputs):
        self._weights = np.random.randn(num_neurons, num_inputs)
        self._biases = np.random.randn(num_neurons)
        # Gradients of output with respect to input/weights/biases
        self._grad_inputs = 0
        self._grad_weights = 0
        self._grad_biases = 0
        # Gradients of weights and biases
        self._weight_grad = 0
        self._biases_grad = 0
        self._next_layer = None

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

    @property
    def weight_grad(self):
        return self._weight_grad

    @property
    def biases_grad(self):
        return self._biases_grad


class ReLuLayer(Layer):
    """Layer with ReLu activated neurons"""
    def forward_pass(self, inputs):
        """Calculate the output of this layer given the input
        Also save gradients of output wrt input, weights and biases"""
        outputs = np.maximum(0, np.dot(self._weights, inputs) + self._biases)
        self._grad_inputs = self._weights
        self._grad_inputs[outputs == 0] = 0
        self._grad_weights = inputs[np.newaxis, :] * np.ones(self._weights.shape)
        self._grad_weights[outputs == 0] = 0
        self._grad_biases = np.ones(self._grad_weights.shape)
        self._grad_biases[outputs == 0] = 0
        return outputs

    def back_propagate(self, gradients_in):
        """Propagate gradients backward through the layer and save weight/bias gradients
        :param gradients_in: vector of gradients (dLoss/dy) heading into each neuron
        :return grad_inputs: vector of gradients heading out of each neuron (dLoss/dx = dLoss/dy * dy/dx)"""
        self._weight_grad = gradients_in[:, np.newaxis] * self._grad_inputs  # dLoss/dw = dLoss/dy * dy/dw
        self._biases_grad = gradients_in[:, np.newaxis] * self._grad_biases  # dLoss/db = dLoss/dy * dy/db
        grad_inputs = gradients_in[:, np.newaxis] * self._grad_weights
        return grad_inputs
