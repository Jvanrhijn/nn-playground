import unittest
import numpy as np
import matplotlib.pyplot as plt
import src.layers as ly
import src.network as net
import src.optim as opt
import src.models as mod


class Test(unittest.TestCase):

    def test_layer(self):
        layer = ly.ReLuLayer(10, 5)
        inp = np.random.random(5)
        layer.forward_pass(inp)
        grad_in = np.random.random(10)
        layer.back_propagate(grad_in)

        network = net.NeuralNetwork(5, 3, 1, 10, ly.ReLuLayer, mod.mse)
        output = network.forward_pass(inp)
        correct_output = np.array([1, 2, 3])
        cost, cost_grad = network.cost(output, correct_output)
        network.back_prop(cost_grad)

    def test_train_net(self):
        network = net.NeuralNetwork(2, 2, 1, 10, ly.TanhLayer, mod.mse)

        train_data = np.random.random(size=(10, 2))
        train_outputs = np.zeros(train_data.shape[0])
        for idx in range(len(train_outputs)):
            if train_data[idx, 0] > train_data[idx, 1]:
                train_outputs[idx] = 0
            else:
                train_outputs[idx] = 1
        output = np.zeros(train_data.shape)
        for idx, example in enumerate(train_data):
            output[idx, :] = network.forward_pass(example)
        correct = 0
        for idx, entry in enumerate(output):
            if entry.argmax() == train_outputs[idx]:
                correct += 1
        start_acc = correct / train_data.shape[0]

        nesterov = opt.NAGOptimizer(0.01, 0.95, network)
        costs = network.train(train_data, train_outputs, 10000, nesterov,
                              quiet=False, save=True)

        outputs = np.zeros(train_data.shape)
        output = np.zeros(train_data.shape)
        for idx, example in enumerate(train_data):
            output[idx, :] = network.forward_pass(example)
        correct = 0
        for idx, entry in enumerate(output):
            if entry.argmax() == train_outputs[idx]:
                correct += 1
        acc = correct / train_data.shape[0]
        print("Start accuracy: {}".format(start_acc))
        print("Final accuracy: {}".format(acc))

        fig, ax = plt.subplots(1)
        ax.semilogy(costs)
        ax.grid()
        ax.set_xlabel("Epoch"), ax.set_ylabel("Objective function")
        plt.show()


