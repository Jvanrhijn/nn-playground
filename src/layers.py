import numpy as np
from src.neurons import Neuron


class Layer:
    """Layer of a fully connected neural network"""
    def __init__(self, *neurons, activator=None):
        self._neurons = list(neurons)
        self._weights = np.array([neuron.weights for neuron in self._neurons])
        if activator is None:
            assert neurons is not None
            self._activator = neurons[0].activation
        else:
            self._activator = activator
        self._next_layer = None
        self.momentum = 0

    def init_random(self, num_neurons, weights_per_neuron):
        self._neurons = [Neuron(self._activator, weights_per_neuron).init_random() for _ in range(num_neurons)]
        self._weights = np.array([neuron.weights for neuron in self._neurons])
        return self

    def forward_pass(self, inputs):
        """Pass inputs through the network"""
        outputs = np.zeros(len(self._neurons))
        for idx, neuron in enumerate(self._neurons):
            outputs[idx] = neuron.activate(inputs)
        return outputs

    def back_propagate(self, gradients_in):
        """Back propagate gradients through the neurons in this layer"""
        gradients_inputs, gradients_weights = np.zeros(self._weights.shape), np.zeros(self._weights.shape)
        for idx, neuron in enumerate(self._neurons):
            grad_inputs, grad_weights = neuron.back_prop(gradients_in.T[idx])
            gradients_inputs[idx, :] = grad_inputs
            gradients_weights[idx, :] = grad_weights
        return gradients_inputs, gradients_weights

    def connect_next(self, next_layer):
        self._next_layer = next_layer

    @property
    def next_layer(self):
        return self._next_layer

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        assert new_weights.shape == self._weights.shape
        for idx, neuron in enumerate(self._neurons):
            neuron.weights = new_weights[idx, :]
        self._weights = new_weights
