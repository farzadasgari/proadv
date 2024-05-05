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
    """
    Calculate the first and second order derivatives of the input data.

    Parameters
    ------
    c (numpy.ndarray): Input data.

    Returns
    ------
    dc (numpy.ndarray): First order derivative of the input data.
    dc2 (numpy.ndarray): Second order derivative of the input data.
    """
    dc = np.gradient(c)
    dc2 = np.gradient(dc)
    return dc, dc2


def _rotation(c, dc, dc2, theta):
    """
    Rotate the input data based on the specified angle theta.

    Parameters
    ----------
    c (numpy.ndarray): Input data.
    dc (numpy.ndarray): First order derivative of the input data.
    dc2 (numpy.ndarray): Second order derivative of the input data.

    Returns
    -------
    x (numpy.ndarray): Rotated X-axis data.
    y (numpy.ndarray): Rotated Y-axis data.
    z (numpy.ndarray): Rotated Z-axis data.
    """
    if theta == 0:
        x = c.copy()
        y = dc.copy()
        z = dc2.copy()
    else:
        rotation_matrix = np.around(np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]), 4)
        x = np.around(c * rotation_matrix[0, 0] + dc * rotation_matrix[0, 1] + dc2 * rotation_matrix[0, 2], 4)
        y = np.around(c * rotation_matrix[1, 0] + dc * rotation_matrix[1, 1] + dc2 * rotation_matrix[1, 2], 4)
        z = np.around(c * rotation_matrix[2, 0] + dc * rotation_matrix[2, 1] + dc2 * rotation_matrix[2, 2], 4)
    return x, y, z


def _parameters(x, y, z):
    lambda_ = np.around(np.sqrt(2 * np.log(x.size)), 4)
    a = np.around(lambda_ * np.nanstd(x), 4)
    b = np.around(lambda_ * np.nanstd(y), 4)
    c = np.around(lambda_ * np.nanstd(z), 4)
    return a, b, c
