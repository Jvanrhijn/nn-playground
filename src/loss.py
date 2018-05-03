import numpy as np


def svm_loss(outputs, correct_outputs, reg=0):
    loss = np.mean(np.sum(np.maximum(0, outputs - correct_outputs + 1), axis=0))
    #loss_grad =
    return None


def mse_loss(outputs, correct_outputs):
    """Return loss and loss gradient with respect to outputs"""
    loss = np.mean(0.25*np.sum((correct_outputs - outputs)**2, axis=0))
    loss_grad = np.mean(-0.5*np.sum(outputs*(correct_outputs - outputs)))
    return loss, loss_grad


def mae_loss(outputs, correct_outputs, reg=0):
    loss = 0.25*np.sum(abs(correct_outputs - outputs))
    loss_grad = 0


def softmax_loss(outputs, correct_outputs, reg=0):
    #loss =
    pass
