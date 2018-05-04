import numpy as np
from src.util import sigmoid


class Layer:
    """Layer of a fully connected neural network"""
    def __init__(self, num_neurons, num_inputs):
        self._num_neurons = num_neurons
        self._weights = 1*np.random.randn(num_neurons, num_inputs)
        self._biases = 1*np.random.randn(num_neurons)
        # Gradients of output with respect to input/weights/biases
        self._grad_inputs = 0
        self._grad_weights = 0
        self._grad_biases = 0
        # Gradients of weights and biases
        self._weight_grad = np.zeros(self._weights.shape)
        self._biases_grad = np.zeros(self._biases.shape)
        self._next_layer = None

    def forward_pass(self, inputs):
        """Pass inputs through the network"""
        pass

    def connect_next(self, next_layer):
        pass

    def back_propagate(self, gradient_in):
        """Propagate gradients backward through the layer and save weight/bias gradients
        :param gradients_in: vector of gradients (dLoss/dy) heading into each neuron
        :return grad_inputs: vector of gradients heading out of each neuron (dLoss/dx = dLoss/dy * dy/dx)"""
        grad_inputs = self._grad_inputs.T.dot(gradient_in)
        self._weight_grad = gradient_in[:, np.newaxis] * self._grad_weights
        self._biases_grad = self._grad_biases.T.dot(gradient_in)
        return grad_inputs

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

    @property
    def num_neurons(self):
        return self._num_neurons


class ReLuLayer(Layer):
    """Layer with ReLu activated neurons"""
    def forward_pass(self, inputs):
        """Calculate the output of this layer given the input
        Also save gradients of output wrt input, weights and biases"""
        outputs = np.maximum(0, np.dot(self._weights, inputs) + self._biases)
        self._grad_inputs = self._weights
        self._grad_inputs[outputs == 0] = 0
        self._grad_weights = inputs[np.newaxis, :] * np.ones(self._grad_inputs.shape)
        self._grad_weights[outputs == 0] = 0
        self._grad_biases = np.ones(self.num_neurons)
        self._grad_weights[outputs == 0] = 0
        return outputs


class LinearLayer(Layer):
    """Linear classifier layer e.g. for output layer"""
    def forward_pass(self, inputs):
        outputs = np.dot(self._weights, inputs) + self._biases
        self._grad_inputs = self._weights
        self._grad_weights = inputs[np.newaxis, :] * np.ones(self._weights.shape)
        self._grad_biases = np.ones(self.num_neurons)
        return outputs


class SigmoidLayer(Layer):
    """Sigmoid activated layer"""
    def forward_pass(self, inputs):
        outputs = sigmoid(np.dot(self._weights, inputs) + self._biases)
        sigmoid_grad = outputs * (1 - outputs)
        self._grad_inputs = sigmoid_grad[np.newaxis, :].T * self._weights
        self._grad_weights = sigmoid_grad[np.newaxis, :].T * (inputs[np.newaxis, :] * np.ones(self._grad_inputs.shape))
        self._grad_biases = np.ones(self._num_neurons) * sigmoid_grad
        return outputs


class TanhLayer(Layer):
    """tanh activated layer"""
    def forward_pass(self, inputs):
        outputs = np.tanh(np.dot(self._weights, inputs) + self._biases)
        tanh_grad = 1 - outputs**2
        self._grad_inputs = tanh_grad[np.newaxis, :].T * self._weights
        self._grad_weights = tanh_grad[np.newaxis, :].T * (inputs[np.newaxis, :] * np.ones(self._grad_inputs.shape))
        self._grad_biases = np.ones(self._num_neurons) * tanh_grad
        return outputs
