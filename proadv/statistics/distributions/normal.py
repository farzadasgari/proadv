import numpy as np


def cdf(array):
    from math import erf, erfc
    array_cdf = np.copy(array)
    if array_cdf.size == 0:
        raise ValueError("cannot calculate PDF with empty array.")
    if np.isnan(array_cdf).any():
        raise TypeError('array cannot contain nan values.')
    np_sqrt = 1.0 / np.sqrt(2)
    array_ns = array_cdf * np_sqrt
    absolute_value = np.fabs(array_ns)
    j = 0
    for i in absolute_value:
        if i < np_sqrt:
            array_cdf[j] = 0.5 + 0.5 * erf(array_ns[j])
        else:
            y = 0.5 * erfc(i)
            if array_ns[j] > 0:
                array_cdf[j] = 1.0 - y
            else:
                array_cdf[j] = y
        j += 1

    return array_cdf


def pdf(array, std=1, mean=0):
    array = np.copy(array)
    if array.size == 0:
        raise ValueError("cannot calculate PDF with empty array")
    if np.isnan(array).any():
        raise TypeError('array cannot contain nan values.')
    x = (-0.5 * np.log(2 * np.pi)) - np.log(std)
    y = np.power(array - mean, 2) / (2 * (std * std))
    array_pdf = np.exp(x - y)
    return array_pdf
