import numpy as np


def svm_loss(outputs, correct_outputs, reg=0):
    for idx, output in enumerate(outputs):
        loss = np.maximum(0, output - correct_outputs + 1)
    return