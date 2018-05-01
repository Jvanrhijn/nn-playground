import numpy as np
from src.layers import Layer


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, output_size, num_hidden, neurons_per_hidden, activator, cost_function):
        self._cost_function = cost_function
        self._layers = [Layer(activator=activator).init_random(neurons_per_hidden, input_size)
                        for _ in range(num_hidden)]
        self._layers[0].init_random(neurons_per_hidden, input_size)
        output_layer = Layer(activator=activator).init_random(output_size, neurons_per_hidden)
        self._layers.append(output_layer)
        self._input_size = input_size
        self._output_size = output_size
        for idx in range(0, num_hidden): # Connect all layers
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
            cur_layer = cur_layer.next_layer
        return outputs

    def save_weights(self):
        pass

    def set_weights(self, weights_list):
        """Set weights from list, format of weight list:
        weight_list[0] = np.ndarray of weights in first hidden layer,
        weight_list[1] = np.ndarray of weights in second hidden layer,
        etc"""
        assert len(weights_list) == len(self._layers)
        for layer_num, weight_mat in enumerate(weights_list):
            self._layers[layer_num].weights = weight_mat

    def cost(self, outputs):
        return self._cost_function(outputs)
