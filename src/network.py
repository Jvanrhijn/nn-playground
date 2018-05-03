import numpy as np
import src.layers as ly


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, output_size, num_hidden, neurons_per_hidden, layer_type, cost_function):
        self._input_size = input_size
        self._output_size = output_size
        self._num_hidden = num_hidden
        self._neurons_per_hidden = neurons_per_hidden  # May be vector of different layer sizes
        self._cost_function = cost_function
        self._layer_type = layer_type
        # Generate hidden layers
        self._layers = [layer_type(neurons_per_hidden, input_size)]
        for idx in range(num_hidden-1):
            self._layers.append(layer_type(neurons_per_hidden, self._layers[idx].num_neurons))
        self._layers.append(ly.LinearLayer(output_size, self._layers[-1].num_neurons))

    def train(self, train_data, train_labels, num_epochs, learn_rate, optimizer, quiet=True, save=False):
        """Train the neural network on the given training data set"""
        pass

    def forward_pass(self, input_data):
        """Pass an input through the network"""
        for layer in self._layers:
            input_data = layer.forward_pass(input_data)
        return input_data

    def save_weights(self):
        pass

    def set_weights(self, weights_list):
        for idx, layer in enumerate(self._layers):
            layer.weights = weights_list[idx]

    def set_biases(self, bias_list):
        for idx, layer in enumerate(self._layers):
            layer.biases = bias_list[idx]

    def cost(self, outputs, correct_outputs):
        pass

    def back_prop(self, cost_grad):
        pass
