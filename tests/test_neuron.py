import unittest
import matplotlib.pyplot as plt
import numpy as np
import src.neurons as nr
import src.activators as act
import src.layers as ly
import src.network as nt
import src.loss as ls


class TestNeuron(unittest.TestCase):

    def test_activate(self):
        neuron = nr.Neuron(act.re_lu, 3)
        neuron.weights = np.array([1, 1, 1])
        inputs = np.array([-0.5, 0.2, 0.4])
        self.assertAlmostEqual(neuron.activate(inputs), 0.1)

    def test_backprop(self):
        neuron = nr.Neuron(act.re_lu, 3)
        neuron.weights = np.array([1, 1, 1])
        inputs = np.array([-0.5, 0.2, 0.4])
        neuron.activate(inputs)
        gradient_in = 1.
        grad_inputs, grad_weights = neuron.back_prop(gradient_in)
        np.testing.assert_array_almost_equal(grad_inputs, np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(grad_weights, np.array([-0.5, 0.2, 0.4]))


class TestLayer(unittest.TestCase):

    def test_forward_pass(self):
        neuron_a = nr.Neuron(act.re_lu, 3)
        neuron_b = nr.Neuron(act.re_lu, 3)
        neuron_c = nr.Neuron(act.re_lu, 3)
        layer = ly.Layer(neuron_a, neuron_b, neuron_c)
        for neuron in layer._neurons:
            neuron.weights = np.ones(3)
        neuron_c.weights = np.ones(3)*2
        inp = np.array([1, 2, 3])
        output = layer.forward_pass(inp)
        expect_out = np.array([6, 6, 12])
        np.testing.assert_array_almost_equal(output, expect_out)
        inp = np.array([-inp[0], inp[1], inp[2]])
        output = layer.forward_pass(inp)
        expect_out = np.array([4, 4, 8])
        np.testing.assert_array_almost_equal(output, expect_out)

    def test_backprop(self):
        neuron_a = nr.Neuron(act.re_lu, 3)
        neuron_b = nr.Neuron(act.re_lu, 3)
        neuron_c = nr.Neuron(act.re_lu, 3)
        layer = ly.Layer(neuron_a, neuron_b, neuron_c)
        for neuron in layer._neurons:
            neuron.init_random()
        gradients_in = np.ones(3)
        inputs = np.array([1, 2, -10])
        layer.forward_pass(inputs)
        grads_inp, grads_weights = layer.back_propagate(gradients_in)
        self.assertEqual(len(grads_inp), 3)
        self.assertEqual(len(grads_weights), 3)


class TestNetwork(unittest.TestCase):

    def test_forward_pass(self):
        network = nt.NeuralNetwork(2, 2, 1, 3, act.re_lu, None)
        network.set_weights([np.array([[0.1, 0.2], [0.4, 0.5], [-0.1, -0.3]]),
                             np.array([[1, 2, 3], [4, 5, 6]])])
        input_data = np.array([1, 2])
        expect_out = np.array([3.3, 9.0])
        output = network.forward_pass(input_data)
        np.testing.assert_array_almost_equal(expect_out, output)

    def test_loss_mse(self):
        network = nt.NeuralNetwork(2, 2, 1, 3, act.re_lu, ls.mse_loss)
        network.set_weights([np.array([[0.1, 0.2], [0.4, 0.5], [-0.1, -0.3]]),
                             np.array([[1, 2, 3], [4, 5, 6]])])
        input_data = np.array([1, 2])
        output = network.forward_pass(input_data)
        correct_out = np.array([1.1, 2.2])
        cost = 12.77
        cost_grad = 34.23
        self.assertAlmostEqual(cost, network.cost(output, correct_out)[0])
        self.assertAlmostEqual(cost_grad, network.cost(output, correct_out)[1])

    def test_back_prop(self):
        network = nt.NeuralNetwork(2, 2, 1, 3, act.re_lu, ls.mse_loss)
        network.set_weights([np.array([[0.1, 0.2], [0.4, 0.5], [-0.1, -0.3]]),
                             np.array([[1, 2, 3], [4, 5, 6]])])
        input_data = np.array([1, 2])
        output = network.forward_pass(input_data)
        correct_out = np.array([1.1, 2.2])
        cost_grad = network.cost(output, correct_out)[1]
        weight_grads = network.back_prop(cost_grad)
        self.assertEqual(len(weight_grads), 2)
        self.assertEqual(weight_grads[0].shape, (2, 3))
        self.assertEqual(weight_grads[1].shape, (3, 2))

    def test_train(self):
        network = nt.NeuralNetwork(2, 2, 1, 100, act.re_lu,
                                   ls.mse_loss)
        train_data = np.random.random(size=(100, 2))
        train_outputs = np.array([[train_data[i, 0] > train_data[i, 1], train_data[i, 0] < train_data[i, 1]] for i in range(train_data.shape[0])])*1
        outputs = np.zeros(train_data.shape)
        for idx, inp in enumerate(train_data):
            out = network.forward_pass(inp)
            outputs[idx, out.argmax()] = 1
        start_acc = (outputs == train_outputs).sum()/(train_outputs.shape[0]*train_outputs.shape[1])

        costs = network.train(train_data, train_outputs, 4000, 0.1, quiet=False, save=True, reg=0, mom_par=0.5)

        outputs = np.zeros(train_data.shape)
        for idx, inp in enumerate(train_data):
            out = network.forward_pass(inp)
            outputs[idx, out.argmax()] = 1
        print("Start accuracy: {}".format(start_acc))
        print("Final accuracy: {}".format((outputs == train_outputs).sum()/(train_outputs.shape[0]*train_outputs.shape[1])))

        fig, ax = plt.subplots(1)
        ax.plot(costs)
        ax.set_xlabel("Epoch"), ax.set_ylabel("Objective function")
        plt.show()


