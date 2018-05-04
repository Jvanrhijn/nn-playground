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


def ce(outputs, correct_outputs):
    cost = (-(correct_outputs*np.log(outputs) + (1 - correct_outputs)*np.log(1 - outputs))).sum()
    cost_grad = (outputs - correct_outputs)/(outputs*(1 - outputs))
    return cost, cost_grad


def expc(outputs, correct_outputs, param):
    cost = param * np.exp(1/param*((outputs - correct_outputs)**2).sum())
    cost_grad = 2/param * (outputs - correct_outputs) * cost
    return cost, cost_grad
