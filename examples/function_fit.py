"""Shows how to use a single-input, single-output neural network to approximate a function"""
import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod

np.random.seed(0)


def linear_function(a, b, x):
    return a*x + b


def quadratic_function(a, b, c, x):
    return a*x**2 + b*x + c


num_hidden = 2
neurons_per_hidden = 10
input_size = 1
output_size = 1
learning_rate = 0.001
mom_par = 0.5

epochs = 10000

network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.ReLuLayer, mod.mse)

optimizer = opt.NAGOptimizer(learning_rate, mom_par, network)
gd_optimizer = opt.GDOptimizer(learning_rate)

# Generate training data
training_in = np.linspace(-1, 1, 10)
training_out = linear_function(1, 0, training_in)
training_input = np.array([[data_point] for data_point in training_in])
training_output = np.array([[data_point] for data_point in training_out])

costs = network.train(training_input, training_output, epochs, optimizer, quiet=False, save=True)

# Validate training set
outputs = np.zeros(len(training_input))
for idx in range(len(outputs)):
    outputs[idx] = network.forward_pass(training_input[idx])[0]

x = np.linspace(-1, 1, 1000)
fix, ax = plt.subplots(2)
ax[0].plot(training_in, outputs, label='network')
ax[0].plot(training_in, training_out, label='training')
ax[0].legend()
ax[1].plot(costs)
ax[1].set_ylabel("Cost function")
ax[1].set_xlabel("Epoch")
ax[0].grid(), ax[1].grid()
plt.show()
