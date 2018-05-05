import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod


# Data set: set of (x, y) coordinates in [0, 1] X [0, 1]
# If (x, y) lies above the line 0.5*(1 + sin(10x)), point is red (0),
# else the point should be colored blue
data_size = 1000
train_data = np.random.random((data_size, 2))
colors = {0: 'red', 1: 'blue'}
train_labels = np.array([0 if X[1] > 0.5*(1 + np.sin(X[0])) else 1
                         for X in train_data])

# Set up the neural network
input_size = 2
output_size = 2
num_hidden = 1
neurons_per_hidden = 10
epochs = 100

learn_rate = 0.01

network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.SigmoidLayer, mod.svm)
optimizer = opt.GDOptimizer(learn_rate)

network.train(train_data, train_labels, epochs, optimizer, quiet=False)

