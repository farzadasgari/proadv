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


def _accumarray(subs, vals, sz):
    accum = np.zeros(sz, dtype=vals.dtype)
    for i, sub in enumerate(subs):
        accum[tuple(sub)] += vals[i]
    return accum


def _histogram(trans, grid):
    rows, cols = trans.shape
    bins = np.zeros((rows, cols), dtype=int)
    hist = np.linspace(0, 1, grid + 1)
    for i in range(cols):
        bins[:, i] = np.digitize(trans[:, i], hist, 1)
        bins[:, i] = np.minimum(bins[:, i], grid - 1)
    binned_data = _accumarray(bins, np.ones(rows), (grid,) * cols) / rows
    return binned_data


def _discrete_cosine_1d(data, weight):
    reordered = np.vstack((data[::2, :], data[::-2, :]))
    transform = np.real(weight * np.fft.fft(reordered))
    return transform


def _discrete_cosine_2d(data):
    rows, columns = data.shape
    if rows != columns:
        raise ValueError('Data shape must be square')
    indices = np.arange(1, rows)
    w = np.concatenate(([1], 2 * np.exp(-1j * indices * np.pi / (2 * rows))))
    weight = np.tile(w[:, np.newaxis], (1, columns))
    discrete = _discrete_cosine_1d(_discrete_cosine_1d(data, weight).T, weight).T
    return discrete


def _k(s_indices):
    step = 2
    index_array = np.arange(start=1, stop=2 * s_indices - 1 + 0.1 * step, step=step)
    return (-1) ** s_indices * np.prod(index_array) / np.sqrt(2 * np.pi)
