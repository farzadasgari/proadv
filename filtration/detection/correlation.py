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
