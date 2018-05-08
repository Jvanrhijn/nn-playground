import numpy as np


def mse(outputs, correct_outputs):
    cost = 0.25*((outputs - correct_outputs)**2).sum()
    cost_grad = 0.5*(outputs - correct_outputs)
    return cost, cost_grad


def svm(outputs, correct_label):
    terms = np.maximum(outputs - outputs[correct_label] + 1, 0)
    terms[correct_label] = 0
    cost = terms.sum()
    cost_grad = np.ones(len(outputs))
    cost_grad[terms == 0] = 0
    return cost, cost_grad


def expc(outputs, correct_outputs, param):
    cost = param * np.exp(1/param*((outputs - correct_outputs)**2).sum())
    cost_grad = 2/param * (outputs - correct_outputs) * cost
    return cost, cost_grad


def ce(outputs, correct_label, offset=0):
    sum = np.exp(outputs - max(outputs)).sum()
    cost = -np.log(np.exp(outputs[correct_label] - max(outputs)) / sum)
    cost_grad = np.exp(outputs - max(outputs)) / sum
    return cost, cost_grad
