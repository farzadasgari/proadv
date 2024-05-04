import numpy as np


def calculate_rho(x, y, theta, a, b):
    """
    Calculate rho value for poincare mapping base detection algorithms.

    Parameters
    ------
        x (numpy.ndarray): First component.
        y (numpy.ndarray): Second component.
        theta (float): Angle between velocity components.
        a (float): Coefficient 'a' used in velocity correlation.
        b (float): Coefficient 'b' used in velocity correlation.

    Returns
    ------
        rho (numpy.ndarray): Rho value calculated for velocity correlation.
    """
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = y * np.cos(theta) - x * np.sin(theta)
    return (xp / a) ** 2 + (yp / b) ** 2
