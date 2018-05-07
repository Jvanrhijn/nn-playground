import numpy as np
import src.layers as ly


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, output_size, num_hidden, neurons_per_hidden, layer_type, cost):
        self._layer_type = layer_type
        self.cost = cost
        self.cost_grad = 0
        # Generate hidden layers
        self._layers = [layer_type(neurons_per_hidden, input_size)]
        for idx in range(num_hidden-1):
            self._layers.append(layer_type(neurons_per_hidden, self._layers[idx].num_neurons))
        self._layers.append(ly.LinearLayer(output_size, self._layers[-1].num_neurons))

    def train(self, train_data, train_output, num_epochs, optimizer, quiet=True, save=False):
        """Train the neural network on the given training data set"""
        if save:
            costs = np.zeros(num_epochs)
        for epoch in range(num_epochs):
            total_cost = 0
            for idx, example in enumerate(train_data):
                output = self.forward_pass(example)
                cost, cost_grad = self.cost(output, train_output[idx])
                self.cost_grad = cost_grad
                total_cost += cost
                self.back_prop(cost_grad)  # Stochastic gradient descent or variants
                optimizer.optimize(self)
            if save:
                costs[epoch] = total_cost / train_data.shape[0]
            if not quiet:
                print("Epoch: {0} | Cost: {1}".format(epoch, total_cost/train_data.shape[0]))
        if save:
            return costs

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

    def back_prop(self, cost_grad):
        weights_grads = []
        bias_grads = []
        grad_in = cost_grad
        for layer in reversed(self._layers):
            grad_in = layer.back_propagate(grad_in)
            weights_grads.append(layer.weight_grad)
            bias_grads.append(layer.biases_grad)

    @property
    def layers(self):
        return self._layers
