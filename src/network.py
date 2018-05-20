import numpy as np
import src.layers as ly


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, output_size, num_hidden, neurons_per_hidden, layer_type, cost, h_et_al=False):
        self._layer_type = layer_type
        self.cost = cost
        self.cost_grad = 0
        # Generate hidden layers
        self._layers = [layer_type(neurons_per_hidden, input_size)]
        # Initialize weights
        if h_et_al:
            init_fact = np.sqrt(2/input_size)
        else:
            init_fact=0.01
        for idx in range(num_hidden-1):
            self._layers.append(layer_type(neurons_per_hidden, self._layers[idx].num_neurons, init_fact=init_fact))
            if h_et_al:
                init_fact = np.sqrt(2/self._layers[-1].num_neurons)
        self._layers.append(ly.LinearLayer(output_size, self._layers[-1].num_neurons))

    def train(self, train_data, train_output, num_epochs, optimizer, quiet=True, save=False, reg=0):
        """Train the neural network on the given training data set"""
        if save:
            costs = np.zeros(num_epochs)
        for epoch in range(num_epochs):
            total_cost = 0
            train_data, train_output = self.shuffle_train_data(train_data, train_output)
            for idx, example in enumerate(train_data):
                example = example # stochastic GD
                output = self.forward_pass(example)
                cost, cost_grad = self.cost(output, train_output[idx])
                if reg != 0:
                    for layer in self._layers:
                        cost += 0.5*reg*np.sum(layer.weights**2)
                self.cost_grad = cost_grad
                total_cost += cost
                self.back_prop(cost_grad, reg=reg)  # Stochastic gradient descent or variants
                optimizer.optimize(self)
            if save:
                costs[epoch] = total_cost / train_data.shape[0]
            if not quiet:
                print("Epoch: {0} | Cost: {1}".format(epoch, total_cost/train_data.shape[0]))
        if save:
            return costs
        else:
            return None

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

    def back_prop(self, cost_grad, reg=0):
        weights_grads = []
        bias_grads = []
        grad_in = cost_grad
        for layer in reversed(self._layers):
            grad_in = layer.back_propagate(grad_in)
            weights_grads.append(layer.weight_grad)
            bias_grads.append(layer.biases_grad)
            if reg != 0:
                layer.weights -= reg*layer.weights

    @property
    def layers(self):
        return self._layers

    @staticmethod
    def shuffle_train_data(train_data, train_output):
        assert len(train_data) == len(train_output)
        permutation = np.random.permutation(len(train_data))
        return train_data[permutation], train_output[permutation]
