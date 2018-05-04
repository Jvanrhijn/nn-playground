"""Comparison of different optimization algorithms"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod


np.random.seed(0)


def demo(func, network, optimizer, training_in, func_name, plot_title=None):
    # Generate training data
    training_out = func(training_in)
    training_input = np.array([[data_point] for data_point in training_in])
    training_output = np.array([[data_point] for data_point in training_out])

    print("Training network to fit: %s" % func_name)
    costs = network.train(training_input, training_output, epochs, optimizer, quiet=True, save=True)
    print("Starting cost: {0}\n""Final cost: {1}\n".format(costs[0], costs[-1]))

    # Validate training set using points between training input points
    validation_in = np.linspace(training_in[0], training_in[-1], 10*len(training_in))
    validation_input = np.array([[data_point] for data_point in validation_in])
    outputs = np.zeros(len(validation_in))
    for idx in range(len(outputs)):
        outputs[idx] = network.forward_pass(validation_input[idx])[0]

    fig, ax = plt.subplots(2)
    if plot_title is not None:
        ax[0].set_title(plot_title)
    ax[0].plot(validation_in, outputs, label='network')
    ax[0].plot(training_in, training_out, '.', label='training')
    ax[0].legend()
    ax[1].semilogy(costs)
    ax[1].set_ylabel("Cost function")
    ax[1].set_xlabel("Epoch")
    ax[0].grid(), ax[1].grid()
    return fig, ax


def gaussian_function(a, b, c, x):
    return a*np.exp(-((x - b)/(2*c))**2)


# Set up hyperparameters
learn_rate_sgd = 0.01
learn_rate_nag = 0.01
mom_par = 0.6

neurons_per_hidden = 10
num_hidden = 1
input_size = 1
output_size = 1
epochs = 2000

network_sgd = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.TanhLayer, mod.mse)
network_nag = copy.deepcopy(network_sgd)

optim_sgd = opt.GDOptimizer(learn_rate_sgd)
optim_nag = opt.NAGOptimizer(learn_rate_nag, mom_par, network_nag)

func = lambda x: gaussian_function(1, 0, 0.25, x)

training_in = np.linspace(-1, 1, 20)

demo(func, network_sgd, optim_sgd, training_in, "gaussian function - SGD", plot_title="Stochastic gradient descent")
demo(func, network_nag, optim_nag, training_in, "gaussian function - NAG", plot_title="Nesterov's accelerated GD")

plt.show()

