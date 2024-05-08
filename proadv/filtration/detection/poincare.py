import numpy as np


def calculate_ab(std1, std2, theta, lambda_):
    """
    Calculate 'a' and 'b' coefficients for poincare mapping base detection algorithms.

    Parameters
    ------
        std1 (float): Standard deviation of the first component.
        std2 (float): Standard deviation of the second component.
        theta (float | rad): Angle between components.
        lambda_ (float): Lambda value used in algorithm.

    Returns
    ------
        float: Coefficient 'a' used in poincare map.
        float: Coefficient 'b' used in poincare map.
    """
    r1 = lambda_ * std1
    r2 = lambda_ * std2
    fact = np.cos(theta) ** 4 - np.sin(theta) ** 4
    fa = (r1 ** 2 * np.cos(theta) ** 2 - r2 ** 2 * np.sin(theta) ** 2) / fact
    fb = (r2 ** 2 * np.cos(theta) ** 2 - r1 ** 2 * np.sin(theta) ** 2) / fact
    return np.sqrt(fa), np.sqrt(fb) if fa > 0 and fb > 0 else (1e10, 1e10)


def calculate_rho(x, y, theta, a, b):
    """
    Calculate rho value for poincare mapping base detection algorithms.

    Parameters
    ------
        x (array_like): First component.
        y (array_like): Second component.
        theta (float | rad): Angle between components.
        a (float): Coefficient 'a' used in poincare map.
        b (float): Coefficient 'b' used in poincare map.

    Returns
    ------
        rho (array_like): Rho value calculated for poincare map.
    """
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = y * np.cos(theta) - x * np.sin(theta)
    return (xp / a) ** 2 + (yp / b) ** 2
