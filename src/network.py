import numpy as np
from src.layers import Layer


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, num_layers, neurons_per_layer, activator, cost_function):
        self._cost_function = cost_function
        self._layers = [Layer(activator=activator).init_random(neurons_per_layer, input_size)
                        for _ in range(num_layers)]
        self._layers[0].init_random(neurons_per_layer, input_size)
        self._input_size = input_size
        for idx in range(0, num_layers-1):
            self._layers[idx].connect_next(self._layers[idx+1])

    def train(self, train_data, train_labels, num_epochs):
        """Train the neural network on the given training data set"""
        pass

    def forward_pass(self, input_data):
        """Pass an input through the network"""
        assert len(input_data) == self._input_size
        inputs = input_data
        cur_layer = self._layers[0]
        outputs = None
        while cur_layer is not None:
            outputs = cur_layer.forward_pass(inputs)
            inputs = outputs
            cur_layer = cur_layer.next_layer()
        return outputs

    def save_weights(self):
        pass

    def set_weights(self):
        pass

    def cost(self, outputs):
        return self._cost_function(outputs)
