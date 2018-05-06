import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod

np.random.seed(0)


def sin_sep(x):
    return 0.5*(1 + np.sin(x))


def test_accuracy(network, test_data, separation_line=sin_sep):
    correct = 0
    for test in test_data:
        output = network.forward_pass(test)
        correct_col = "red" if test[1] > separation_line(test[0]) else "blue"
        if correct_col == colors[output.argmax()]:
            correct += 1
    return correct / test_data.shape[0]


# Data set: set of (x, y) coordinates in [0, 1] X [0, 1]
# If (x, y) lies above the line 0.5*(1 + sin(10x)), point is red (0),
# else the point should be colored blue
data_size = 100
train_data = np.random.random((data_size, 2))
colors = {0: 'red', 1: 'blue'}
train_labels = np.array([0 if X[1] > 0.5*(1 + np.sin(X[0])) else 1
                         for X in train_data])

# Set up the neural network
input_size = 2
output_size = 2
num_hidden = 1
neurons_per_hidden = 1000
epochs = 2000

learn_rate = 0.00001
mom_par = 0.5

network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, ly.ReLuLayer, mod.svm)

# Get pre-training accuracy
test_size = 1000
test_data = np.random.random((test_size, 2))
acc_before = test_accuracy(network, test_data)

optimizer = opt.NAGOptimizer(learn_rate, mom_par, network)

costs = network.train(train_data, train_labels, epochs, optimizer, quiet=False, save=True)

acc_after = test_accuracy(network, test_data)
print("Accuracy before: {0}\nAccuracy after: {1}\n".format(acc_before, acc_after))

plt.figure()
plt.semilogy(costs)
plt.grid()
plt.show()

