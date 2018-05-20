import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import src.layers as ly
import src.optim as opt
import src.models as mod

np.random.seed(0)


def sep(x):
    return 0.5*(1 + np.tanh(3*x-1))


def test_accuracy(network, test_data, test_labels):
    correct = 0
    results = np.zeros(len(test_labels))
    for idx, ex in enumerate(test_data):
        pred_class = np.argmax(network.forward_pass(ex))
        results[idx] = pred_class
        if pred_class == test_labels[idx]:
            correct += 1
    accuracy = correct / len(test_labels)
    return accuracy, results


# CS231n spiral dataset
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

num_hidden = 2
neurons_per_hidden = 50
epochs = 1000

learn_rate = 1e-4
window = 0.9
window_sq = 0.999

network = net.NeuralNetwork(D, K, num_hidden, neurons_per_hidden,
                            activation='relu',
                            cost='ce',
                            h_et_al=True)

acc_before = test_accuracy(network, X, y)[0]

costs = network.train(X, y, epochs,
                      optimizer='adam',
                      lr=learn_rate,
                      beta1=window,
                      beta2=window_sq,
                      quiet=False, save=True, reg=1e-5)

acc_after = test_accuracy(network, X, y)[0]
print("Accuracy before: {0}\nAccuracy after: {1}\n".format(acc_before, acc_after))

figure = plt.figure()
ax_data = figure.add_subplot(121)
ax_cost = figure.add_subplot(122)

ax_cost.semilogy(costs)
ax_cost.set_ylabel("Cost function")
ax_cost.set_xlabel("Epoch")

test_size = 1000
test_data = np.random.random((test_size, 2))*2 - 1
results = np.zeros(test_data.shape[0])
colors = {
    0: 'red',
    1: 'blue',
    2: 'green'
}
for idx, point in enumerate(test_data):
    result = np.argmax(network.forward_pass(point))
    ax_data.plot(point[0], point[1], 'o', color=colors[result])

ax_cost.grid()

plt.show()

