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
