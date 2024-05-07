import numpy as np


def _rotation(x, y):
    data_size = x.size
    numerator = data_size * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = data_size * np.sum(x * x) - np.sum(x) * np.sum(x)
    theta = np.arctan2(numerator, denominator)
    return theta


def _scaling(data):
    max_co = data.max(1)
    min_co = data.min(1)
    scale = max_co - min_co
    return max_co, min_co, scale


def _transform(data, max_co, min_co, scale):
    numerator = data.T - np.tile(min_co, (data[0].size, 1))
    denominator = np.tile(scale, (data[0].size, 1))
    transformed_data = numerator / denominator
    return transformed_data
