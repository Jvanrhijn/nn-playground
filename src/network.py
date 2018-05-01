import numpy as np
from src.layers import Layer


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_layer, num_layers, neurons_per_layer, activator):
        self._layers = [Layer(activator).init_random(neurons_per_layer, num_layers)]

    def train(self, train_data, train_labels):
        """Train the neural network on the given training data set"""
        pass

    def classify(self, input_data):
        """Classify an input"""
        pass
