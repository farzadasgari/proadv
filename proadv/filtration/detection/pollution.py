import numpy as np
from proadv.statistics.spread import std


def pollution_rate(x, xfil, k):
    """
    Compute the pollution rate based on the given data and threshold.

    Parameters
    ------
    x (array_like): Input data array.
    xfil (array_like): SSA analyzed data of x.
    k (float): Threshold factor.

    Returns
    ------
    pollution (float): Pollution rate as a percentage.
    """
    pos = np.nonzero(x > xfil + k * std(x))
    neg = np.nonzero(x < -xfil - k * std(x))
    noises = np.concatenate((pos, neg), axis=None)
    pollution = len(noises) / len(x) * 100
    return pollution
