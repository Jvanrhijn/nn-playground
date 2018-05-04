"""Shows how to use a single-input, single-output neural network to approximate some functions"""
import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod

np.random.seed(0)


def demo(func, network, optimizer, training_in, func_name):
    # Generate training data
    training_out = func(training_in)
    training_input = np.array([[data_point] for data_point in training_in])
    training_output = np.array([[data_point] for data_point in training_out])

    print("Training network to fit: %s" % func_name)
    costs = network.train(training_input, training_output, epochs, optimizer, quiet=True, save=True)
    print("Starting cost: {0}\n""Final cost: {1}\n".format(costs[0], costs[-1]))

    # Validate training set
    outputs = np.zeros(len(training_input))
    for idx in range(len(outputs)):
        outputs[idx] = network.forward_pass(training_input[idx])[0]

    fig, ax = plt.subplots(2)
    ax[0].plot(training_in, outputs, label='network')
    ax[0].plot(training_in, training_out, '.', label='training')
    ax[0].legend()
    ax[1].semilogy(costs)
    ax[1].set_ylabel("Cost function")
    ax[1].set_xlabel("Epoch")
    ax[0].grid(), ax[1].grid()
    return fig, ax


def linear_function(a, b, x):
    return a*x + b


def quadratic_function(a, b, c, x):
    return a*x**2 + b*x + c


def gaussian_function(a, b, c, x):
    return a*np.exp(-((x - b)/(2*c))**2)


"""Linear function example"""
num_hidden = 1
neurons_per_hidden = 100
input_size = 1
output_size = 1
learning_rate = 0.0001
mom_par = 0.5

epochs = 500

training_in = np.linspace(-1, 1, 20)

lin_network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.ReLuLayer, mod.mse)

nag_optimizer = opt.NAGOptimizer(learning_rate, mom_par, lin_network)

demo(lambda x: linear_function(-2, 2, x), lin_network, nag_optimizer, training_in, "linear function")


"""Quadratic function example"""
num_hidden = 2
neurons_per_hidden = 100
input_size = 1
output_size = 1
learning_rate = 0.001
mom_par = 0.7

epochs = 500

training_in = np.linspace(-1, 1, 20)

quad_network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.TanhLayer, mod.mse)

nag_optimizer = opt.NAGOptimizer(learning_rate, mom_par, quad_network)

demo(lambda x: quadratic_function(1, 0, 0, x), quad_network, nag_optimizer, training_in, "quadratic function")


"""Gaussian function example"""
num_hidden = 1
neurons_per_hidden = 100
input_size = 1
output_size = 1
learning_rate = 0.001
mom_par = 0.8

epochs = 2000

gauss_network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.TanhLayer, mod.mse)

nag_optimizer = opt.NAGOptimizer(learning_rate, mom_par, gauss_network)

training_in = np.linspace(-1, 1, 20)

demo(lambda x: gaussian_function(1, 0, .25, x), gauss_network, nag_optimizer, training_in, "gaussian function")

plt.show()
