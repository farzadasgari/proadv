import numpy as np


def calculate_parameters(up, vp, wp):
    data_size = up.size
    std_u = np.std(up)
    std_v = np.std(vp)
    std_w = np.std(wp)
    lambda_ = np.sqrt(2 * np.log(data_size))
    return lambda_, std_u, std_v, std_w
