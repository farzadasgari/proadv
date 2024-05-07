import numpy as np


def _rotation(x, y):
    data_size = x.size
    numerator = data_size * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = data_size * np.sum(x * x) - np.sum(x) * np.sum(x)
    theta = np.arctan2(numerator, denominator)
    return theta
