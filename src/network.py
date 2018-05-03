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
        for idx in range(0, num_hidden):  # Connect all layers
            self._layers[idx].connect_next(self._layers[idx+1])

    def train(self, train_data, train_labels, num_epochs, learn_rate, quiet=True, save=False, reg=0, mom_par=0):
        """Train the neural network on the given training data set"""
        if save:
            costs = np.zeros(num_epochs)
        for epoch in range(num_epochs):
            output = np.zeros(train_labels.shape)
            for idx, example in enumerate(train_data):
                output[idx, :] = self.forward_pass(example)
            cost, cost_grad = self.cost(output, train_labels, reg=reg)
            if save:
                costs[epoch] = cost
            if not quiet:
                print("Epoch: {0} | cost: {1}".format(epoch, cost))
            weight_grads = self.back_prop(cost_grad)
            for idx, layer in enumerate(reversed(self._layers)):
                momentum_prev = layer.momentum
                layer.momentum = mom_par*layer.momentum + learn_rate * weight_grads[idx]
                layer.weights -= mom_par*momentum_prev + (1 + mom_par) * layer.momentum
        if save:
            return costs

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

    def cost(self, outputs, correct_outputs, reg=0):
        regularization = 0
        for layer in self._layers:
            regularization += np.sum(layer.weights**2)
        return self._cost_function(outputs, correct_outputs) + reg*regularization

    def back_prop(self, cost_grad):
        """Back propagate through the layers of the network, retrieving the gradient of the cost function with
        respect to all the weights"""
        weight_grads = []
        gradient_in = np.ones(len(self._layers[-1]._neurons))*cost_grad
        for layer in reversed(self._layers):
            grad_inputs, grad_weights = layer.back_propagate(gradient_in)
            weight_grads.append(grad_weights)
            gradient_in = grad_inputs
        return weight_grads
