import numpy as np


def _descript(c, iteration, c_mean):
    if iteration == 1:
        f_mean = np.around(np.nanmean(c), 4)
    else:
        f_mean = np.around(c_mean + np.nanmean(c), 4)
    f = np.around(c - np.nanmean(c), 4)
    return f, f_mean
