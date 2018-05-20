import numpy as np
import src.layers as ly
import src.optim as opt
import src.models as mod


class NeuralNetwork:
    """Implements a fully connected neural network"""
    def __init__(self, input_size, output_size, num_hidden, neurons_per_hidden, activation='relu', cost='ce', h_et_al=False):
        self._layer_type = self._set_activation(activation)
        self.cost = self._set_cost(cost)
        self.cost_grad = 0
        self.optimizer = None
        # Generate hidden layers
        self._layers = [self._layer_type(neurons_per_hidden, input_size)]
        # Initialize weights
        if h_et_al:
            init_fact = np.sqrt(2/input_size)
        else:
            init_fact = 0.01
        for idx in range(num_hidden-1):
            self._layers.append(self._layer_type(neurons_per_hidden, self._layers[idx].num_neurons,
                                                 init_fact=init_fact))
            if h_et_al:
                init_fact = np.sqrt(2/self._layers[-1].num_neurons)
        self._layers.append(ly.LinearLayer(output_size, self._layers[-1].num_neurons))

    def train(self, train_data, train_output, num_epochs,
              optimizer='sgd',
              lr=1e-4,
              mom=0.0,
              gamma=0.9,
              beta1=0.9,
              beta2=0.999,
              nesterov=False,
              quiet=True, save=False, reg=0):
        """Train the neural network on the given training data set"""
        self.set_optimizer(optimizer, lr=lr, mom=mom, gamma=gamma, beta1=beta1, beta2=beta2, nesterov=nesterov)
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
                self.optimizer.optimize(self)
            if save:
                costs[epoch] = total_cost / train_data.shape[0]
            if not quiet:
                print("Epoch: {0} | Cost: {1}".format(epoch, total_cost/train_data.shape[0]))
        if save:
            return costs
        else:
            return None

    def set_optimizer(self, optimizer='sgd', lr=1e-3, mom=0.5, gamma=0.9, beta1=0.9, beta2=0.999, nesterov=False):
        if optimizer == 'sgd':
            self.optimizer = opt.GDOptimizer(lr)
        elif optimizer == 'momentum':
            if nesterov:
                self.optimizer = opt.NAGOptimizer(lr, mom, self)
            else:
                self.optimizer = opt.MomentumOptimizer(lr, mom, self)
        elif optimizer == 'rmsprop':
            self.optimizer = opt.RMSpropOptmizer(lr, gamma, self)
        elif optimizer == 'adagrad':
            self.optimizer = opt.AdaGradOptimizer(lr, self)
        elif optimizer == 'adadelta':
            self.optimizer = opt.AdaDeltaOptimizer(gamma, self)
        elif optimizer == 'adam':
            self.optimizer = opt.AdamOptimizer(lr, beta1, beta2, self)
        else:
            raise ValueError("Optimizer not supported")

    @staticmethod
    def _set_activation(layer_type):
        if layer_type == 'linear':
            return ly.LinearLayer
        elif layer_type == 'relu':
            return ly.ReLuLayer
        elif layer_type == 'tanh':
            return ly.TanhLayer
        elif layer_type == 'sigmoid':
            return ly.SigmoidLayer
        else:
            ValueError("Activation function not supported")

    @staticmethod
    def _set_cost(cost):
        if cost == 'mse':
            return mod.mse
        elif cost == 'svm':
            return mod.svm
        elif cost == 'ce':
            return mod.ce
        elif cost == 'expc':
            return mod.expc

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
        grad_in = cost_grad
        for layer in reversed(self._layers):
            grad_in = layer.back_propagate(grad_in)
            weights_grads.append(layer.weight_grad)
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
