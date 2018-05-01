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

    def init_random(self, num_neurons, weights_per_neuron):
        self._neurons = [Neuron(self._activator, weights_per_neuron) for _ in range(num_neurons)]
        self._weights = np.array([neuron.weights for neuron in self._neurons])

    def forward_pass(self, inputs):
        outputs = np.zeros(len(self._neurons))
        for idx, neuron in enumerate(self._neurons):
            outputs[idx] = neuron.activate(inputs[idx])
        return outputs

    def connect_next(self, next_layer):
        self._next_layer = next_layer

