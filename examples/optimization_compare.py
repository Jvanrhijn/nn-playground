"""Comparison of different optimization algorithms"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod


np.random.seed(0)


def demo(func, network, optimizer, training_in, func_name, plot_title=None, quiet=True):
    # Generate training data
    training_out = func(training_in)
    training_input = np.array([[data_point] for data_point in training_in])
    training_output = np.array([[data_point] for data_point in training_out])

    print("Training network to fit: %s" % func_name)
    costs = network.train(training_input, training_output, epochs, optimizer, quiet=quiet, save=True)
    print("Starting cost: {0}\n""Final cost: {1}\n".format(costs[0], costs[-1]))

    # Validate training set using points between training input points
    validation_in = np.linspace(training_in[0], training_in[-1], 10*len(training_in))
    validation_input = np.array([[data_point] for data_point in validation_in])
    outputs = np.zeros(len(validation_in))
    for idx in range(len(outputs)):
        outputs[idx] = network.forward_pass(validation_input[idx])[0]

    return validation_in, outputs, costs


def gaussian_function(a, b, c, x):
    return a*np.exp(-((x - b)/(2*c))**2)


# Set up hyperparameters
learn_rate_sgd = 2*1e-2

learn_rate_mom = 0.025
mom_par = 0.9

learn_rate_nag = 1e-3
mom_par_nag = 0.9

learn_rate_adagrad = 0.5

window_size_adadelta = 0.999

window_size_rmsprop = 0.9
learn_rate_rmsprop = 0.0075

learn_rate_adam = 0.05
window_grad_adam = 0.9
window_sq_adam = 0.999

neurons_per_hidden = 50
num_hidden = 2
input_size = 1
output_size = 1
epochs = 5000

network_sgd = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden,
                                ly.TanhLayer, mod.mse, h_et_al=True)
network_nag = copy.deepcopy(network_sgd)
network_mom = copy.deepcopy(network_sgd)
network_adagrad = copy.deepcopy(network_sgd)
network_adadelta = copy.deepcopy(network_sgd)
network_rmsprop = copy.deepcopy(network_sgd)
network_adam = copy.deepcopy(network_sgd)

optim_sgd = opt.GDOptimizer(learn_rate_sgd)
optim_mom = opt.MomentumOptimizer(learn_rate_mom, mom_par, network_mom)
optim_nag = opt.NAGOptimizer(learn_rate_nag, mom_par_nag, network_nag)
optim_adagrad = opt.AdaGradOptimizer(learn_rate_adagrad, network_nag)
optim_adadelta = opt.AdaDeltaOptimizer(window_size_adadelta, network_nag)
optim_rmpsprop = opt.RMSpropOptmizer(learn_rate_rmsprop, window_size_rmsprop, network_rmsprop)
optim_adam = opt.AdamOptimizer(learn_rate_adam, window_grad_adam, window_sq_adam, network_adam)

func = lambda x: gaussian_function(1, 0, 0.25, x)

training_in = np.linspace(-1, 1, 50)

x_sgd, y_sgd, costs_sgd = demo(func, network_sgd, optim_sgd, training_in, "gaussian function - SGD",
                               plot_title="Stochastic gradient descent", quiet=True)
x_nag, y_nag, costs_nag = demo(func, network_nag, optim_nag, training_in, "gaussian function - NAG",
                               plot_title="Nesterov's accelerated GD", quiet=True)
x_mom, y_mom, costs_mom = demo(func, network_mom, optim_mom, training_in, "gaussian function - momentum",
                               plot_title="Nesterov's  ", quiet=True)
x_adagrad, y_adagrad, costs_adagrad = demo(func, network_adagrad, optim_adagrad, training_in, "gaussian function - AdaGrad",
                                           plot_title="AdaGrad", quiet=True)
x_adadelta, y_adadelta, costs_adadelta= demo(func, network_adadelta, optim_adadelta, training_in, "gaussian function - AdaDelta",
                                           plot_title="AdaDelta", quiet=True)
x_rmsprop, y_rmsprop, costs_rmsprop = demo(func, network_rmsprop, optim_rmpsprop, training_in, "gaussian function - RMSProp",
                                             plot_title="RMSProp", quiet=True)
x_adam, y_adam, costs_adam = demo(func, network_adam, optim_adam, training_in, "gaussian function - Adam",
                                           plot_title="ADAM", quiet=True)

fig = plt.figure()
ax_fit = fig.add_subplot(121)
ax_cost = fig.add_subplot(122)
ax = [ax_fit, ax_cost]

ax[0].plot(x_sgd, y_sgd, label="SGD")
ax[0].plot(x_mom, y_mom, label="Momentum")
ax[0].plot(x_nag, y_nag, label="NAG")
ax[0].plot(x_adagrad, y_adagrad, label="AdaGrad")
ax[0].plot(x_adadelta, y_adadelta, label="AdaDelta")
ax[0].plot(x_rmsprop, y_rmsprop, label="RMSProp")
ax[0].plot(x_adam, y_adam, label="ADAM")
ax[0].plot(training_in, func(training_in), '.', label="Training points")

ax[1].semilogy(costs_sgd, label="SGD")
ax[1].semilogy(costs_mom, label="Momentum")
ax[1].semilogy(costs_nag, label="NAG")
ax[1].semilogy(costs_adagrad, label="AdaGrad")
ax[1].semilogy(costs_adadelta, label="AdaDelta")
ax[1].semilogy(costs_rmsprop, label="RMSProp")
ax[1].semilogy(costs_adam, label="ADAM")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Cost function")

for axis in ax:
    axis.legend(), axis.grid()

plt.show()

