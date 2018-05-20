import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod

np.random.seed(0)


def sin_sep(x):
    return 0.5*(1 + np.sin(2*np.pi*x))


def parabolic_sep(x):
    return x**2


def tanh_sep(x):
    return 0.5*(1 + np.tanh(3*x-1))
    #return x


def test_accuracy(network, test_data, separation_line=sin_sep):
    correct = 0
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    results = []
    for test in test_data:
        output = network.forward_pass(test)
        results.append(output)
        correct_col = "red" if test[1] > separation_line(test[0]) else "blue" if test[0] > 0.5 else 'green'
        if correct_col == colors[output.argmax()]:
            correct += 1
    return correct / test_data.shape[0], results


# Data set: set of (x, y) coordinates in [0, 1] X [0, 1]
# Color point red, blue or green depending on its location
data_size = 100
train_data = np.random.random((data_size, 2))
train_labels = np.array([0 if X[1] > tanh_sep(X[0]) else 1 if X[0] > 0.5 else 2
                         for X in train_data])

# Set up the neural network
input_size = 2
output_size = 3
num_hidden = 2
neurons_per_hidden = 50
epochs = 5000

learn_rate = 1e-4
window = 0.9
window_sq = 0.999

network = net.NeuralNetwork(input_size, output_size, num_hidden, neurons_per_hidden, activation='relu', cost='ce',
                            h_et_al=True)

# Get pre-training accuracy
test_size = 1000
test_data = np.random.random((test_size, 2))
acc_before = test_accuracy(network, test_data, separation_line=tanh_sep)[0]

optimizer = opt.AdamOptimizer(learn_rate, window, window_sq, network)

costs = network.train(train_data, train_labels, epochs, optimizer='adam',
                      lr=learn_rate,
                      beta1=window,
                      beta2=window_sq,
                      quiet=False, save=True, reg=0)

acc_after, results = test_accuracy(network, test_data, separation_line=tanh_sep)
print("Accuracy before: {0}\nAccuracy after: {1}\n".format(acc_before, acc_after))

figure = plt.figure()
ax_data = figure.add_subplot(121)
ax_cost = figure.add_subplot(122)

ax_cost.semilogy(costs)
ax_cost.set_ylabel("Cost function")
ax_cost.set_xlabel("Epoch")

x_sep = np.linspace(0, 1, 1000)
colors = {0: 'red', 1: 'blue', 2: 'green'}
for point, result in zip(test_data, results):
    color = colors[result.argmax()]
    ax_data.plot(point[0], point[1], 'o', color=color)
ax_data.plot(x_sep, tanh_sep(x_sep), color="black", linewidth=2)
ax_data.plot(np.ones(100)*0.5, np.linspace(0, tanh_sep(0.5), 100), color="black", linewidth=2)
ax_cost.grid()

plt.show()

