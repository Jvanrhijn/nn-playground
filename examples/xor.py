"""Trains a network to emulate the XOR function"""
import numpy as np
import src.network as net


train_data = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
train_labels = np.array([[0], [1], [1], [0]])


network = net.NeuralNetwork(2, 1, 2, 10, cost='mse', h_et_al=True)

network.train(train_data, train_labels, 100, optimizer='nadam',
              lr=1e-3,
              beta1=0.9,
              beta2=0.999)

for ex, ans in zip(train_data, train_labels):
    print("Example: {0}\nNetwork: {1}\n".format(
        ex, int(round(network.forward_pass(ex)[0]))))


