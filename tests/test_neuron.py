import unittest
import numpy as np
import src.layers as ly


class TestLayer(unittest.TestCase):

    def test_forward(self):
        inp = np.array([1, 2, 3])
        weights = np.array([[1, 2, 3], [3, 4, 5]])
        biases = np.array([5, -30])
        expect_out = np.array([19, 0])
        layer = ly.ReLuLayer(2, 2)
        layer.weights = weights
        layer.biases = biases
        grad_inputs = np.array([[1, 2, 3], [0, 0, 0]])
        grad_weights = np.array([[1, 2, 3], [0, 0, 0]])
        grad_biases = np.array([[1, 1, 1], [0, 0, 0]])
        np.testing.assert_array_almost_equal(layer.forward_pass(inp), expect_out)
        np.testing.assert_array_almost_equal(layer.grad_inputs, grad_inputs)
        np.testing.assert_array_almost_equal(layer.grad_weights, grad_weights)
        np.testing.assert_array_almost_equal(layer.grad_biases, grad_biases)

