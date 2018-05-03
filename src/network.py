import numpy as np
from src.layers import Layer


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, output_size, num_hidden, neurons_per_hidden, activator, cost_function):
        pass

    def train(self, train_data, train_labels, num_epochs, learn_rate, optimizer, quiet=True, save=False):
        """Train the neural network on the given training data set"""
        pass

    def forward_pass(self, input_data):
        """Pass an input through the network"""
        pass

    def save_weights(self):
        pass

    def set_weights(self, weights_list):
        pass

    def cost(self, outputs, correct_outputs):
        pass

    def back_prop(self, cost_grad):
        pass
