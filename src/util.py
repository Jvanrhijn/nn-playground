import math
import numpy as np


def sigmoid(x):
    return 1/(np.exp(-x) + 1)


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

