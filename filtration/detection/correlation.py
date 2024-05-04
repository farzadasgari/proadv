import numpy as np


def calculate_parameters(up, vp, wp):
    """
    Calculate parameters required for velocity correlation.

    Parameters
    ------
        up (numpy.ndarray): Array of the first velocity component.
        vp (numpy.ndarray): Array of the second velocity component.
        wp (numpy.ndarray): Array of the third velocity component.

    Returns
    ------
        lambda_ (float): Lambda value used in velocity correlation detection.
        std_u (float): Standard deviation of the longitudinal velocity component.
        std_v (float): Standard deviation of the transverse velocity component.
        std_w (float): Standard deviation of the vertical velocity component.
    """
    from proadv.statistics.spread import std
    data_size = up.size
    std_u = std(up)
    std_v = std(vp)
    std_w = std(wp)
    lambda_ = np.sqrt(2 * np.log(data_size))
    return lambda_, std_u, std_v, std_w


def calculate_ab(std1, std2, theta, lambda_):
    """
    Calculate 'a' and 'b' coefficients for velocity correlation detection.

    Parameters
    ------
        std1 (float): Standard deviation of the first velocity component.
        std2 (float): Standard deviation of the second velocity component.
        theta (float): Angle between velocity components.
        lambda_: Lambda value used in velocity correlation.

    Returns
    ------
        float: Coefficient 'a' used in velocity correlation.
        float: Coefficient 'b' used in velocity correlation.
    """
    r1 = lambda_ * std1
    r2 = lambda_ * std2
    fact = np.cos(theta) ** 4 - np.sin(theta) ** 4
    fa = (r1 ** 2 * np.cos(theta) ** 2 - r2 ** 2 * np.sin(theta) ** 2) / fact
    fb = (r2 ** 2 * np.cos(theta) ** 2 - r1 ** 2 * np.sin(theta) ** 2) / fact
    return np.sqrt(fa), np.sqrt(fb) if fa > 0 and fb > 0 else (1e10, 1e10)
