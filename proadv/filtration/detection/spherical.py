import numpy as np


def _descript(c, iteration, c_mean):
    """
    Calculate the mean value and center the input data.

    Parameters
    ------
    c (numpy.ndarray): Input data.
    iteration (int): Loop counter.
    c_mean (float): Mean value of input data.

    Returns
    ------
    f (numpy.ndarray): Centered input data.
    f_mean (float): Updated mean value.
    """
    if iteration == 1:
        f_mean = np.around(np.nanmean(c), 4)
    else:
        f_mean = np.around(c_mean + np.nanmean(c), 4)
    f = np.around(c - np.nanmean(c), 4)
    return f, f_mean


def _gradients(c):
    dc = np.gradient(c)
    dc2 = np.gradient(dc)
    return dc, dc2
