import numpy as np
from proadv.statistics.spread import std


def pollution_rate(x, xfil, k):
    pos = np.nonzero(x > xfil + k * std(x))
    neg = np.nonzero(x < -xfil - k * std(x))
    noises = np.concatenate((pos, neg), axis=None)
    pollution = len(noises) / len(x) * 100
    return pollution
