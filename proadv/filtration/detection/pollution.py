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
    
    # Find positions where x exceeds the threshold
    pos = np.nonzero(x > xfil + k * std(x))
    
    # Find positions where x is below the negative threshold
    neg = np.nonzero(x < -xfil - k * std(x))
    
    # Concatenate positive and negative noises
    noises = np.concatenate((pos, neg), axis=None)
    
    # Compute pollution rate
    pollution = len(noises) / len(x) * 100
    
    return pollution
