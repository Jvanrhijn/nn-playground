import numpy as np


def mse(outputs, correct_outputs):
    cost = 0.25*((outputs - correct_outputs)**2).sum()
    cost_grad = 0.5*(outputs - correct_outputs)
    return cost, cost_grad


def svm(outputs, correct_outputs):
    terms = np.maximum(outputs - outputs[correct_outputs] + 1, 0)
    terms[correct_outputs] = 0
    cost = terms.sum()
    cost_grad = terms.count_nonzero()
    return cost, cost_grad




