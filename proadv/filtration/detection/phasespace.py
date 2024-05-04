import numpy as np


def calculate_derivatives(c):
    """
    Calculate time-independent first and second order derivatives of the input data.

    Parameters:
        c (numpy.ndarray): Input data.

    Returns:
        dc (numpy.ndarray): First derivative of the input data.
        dc2 (numpy.ndarray): Second derivative of the input data.
    """
    # Initialize arrays for first and second derivatives
    dc = np.zeros_like(c)
    dc2 = np.zeros_like(c)

    # Calculate first derivative
    for i in range(1, len(c) - 1):
        dc[i] = np.around((c[i + 1] - c[i - 1]) / 2, 4)

    # Calculate second derivative
    for i in range(1, len(c) - 1):
        dc2[i] = np.around((dc[i + 1] - dc[i - 1]) / 2, 4)

    return dc, dc2


def calculate_parameters(c, dc, dc2):
    std_c = np.std(c)
    std_dc = np.std(dc)
    std_dc2 = np.std(dc2)
    lambda_ = np.sqrt(2 * np.log(len(c)))
    theta = np.arctan(np.sum(c * dc2) / np.sum(c ** 2))
    a1 = lambda_ * std_c
    b1 = lambda_ * std_dc
    a2 = lambda_ * std_dc
    b2 = lambda_ * std_dc2
    fact = np.cos(theta) ** 4 - np.sin(theta) ** 4
    a3 = np.sqrt(a1 ** 2 * np.cos(theta) ** 2 - b2 ** 2 * np.sin(theta) ** 2) / fact
    b3 = np.sqrt(b2 ** 2 * np.cos(theta) ** 2 - a1 ** 2 * np.sin(theta) ** 2) / fact
    return std_c, std_dc, std_dc2, lambda_, theta, a1, b1, a2, b2, a3, b3
