"""Comparison of different optimization algorithms"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import src.network as net
import matplotlib
font = {'size': 12}

matplotlib.rc('font', **font)


np.random.seed(0)

# CS231n spiral dataset
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N*K, D))  # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j


def train(network, algorithm, epochs=1000,
          lr=1e-3, mom=0.9, gamma=0.9, beta1=0.9, beta2=0.999,
          reg=0.0, nesterov=False):
    costs = network.train(X, y, epochs, optimizer=algorithm,
                          lr=lr,
                          mom=mom,
                          gamma=gamma,
                          beta1=beta1,
                          beta2=beta2,
                          nesterov=nesterov,
                          save=True, quiet=True, reg=reg)
    return costs


# network properties
num_hidden = 1
neurons_per_hidden = 100

network = net.NeuralNetwork(2, K, num_hidden, neurons_per_hidden,
                            activation='relu', cost='ce', h_et_al=True)


network_nag = copy.deepcopy(network)
network_adagrad = copy.deepcopy(network)
network_adadelta = copy.deepcopy(network)
network_rmsprop = copy.deepcopy(network)
network_adam = copy.deepcopy(network)
network_nadam = copy.deepcopy(network)


costs_sgd = train(network, "sgd", lr=1e-2)
costs_nag = train(network_nag, "momentum", nesterov=True, lr=1e-3, mom=0.8)
costs_rmsprop = train(network_rmsprop, "rmsprop", lr=2*1e-3)
costs_adam = train(network_adam, "adam", lr=2*1e-3)
costs_nadam = train(network_nadam, "nadam", lr=2*1e-3)

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.semilogy(costs_sgd, label='SGD')
ax.semilogy(costs_nag, label="NAG")
ax.semilogy(costs_rmsprop, label="RMSProp")
ax.semilogy(costs_adam, label="Adam")
ax.semilogy(costs_nadam, label="Nadam")
ax.legend()
ax.grid()

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy cost")

plt.show()
