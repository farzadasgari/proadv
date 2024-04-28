import numpy as np
from desc import mean #Direct import
from spread import std #Direct import

def skewness(data):
    '''
    Compute the sample skewness of a data set. 

    For normally distributed data, the skewness should be about zero. For
    unimodal continuous distributions, a skewness value greater than zero means
    that there is more weight in the right tail of the distribution. The
    function `skewtest` can be used to determine if the skewness value
    is close enough to zero, statistically speaking.

    Parameters
    -------
    data (np.ndarray): The 1D array of data for which to calculate the Skewness. 

    Returns
    -------
    skewness : ndarray
        The skewness of values along an axis, returning NaN where all values
        are equal.

    References
    -------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section 2.2.24.1

    Examples
    --------
    from scipy.stats import skew
    >>> skew([1, 2, 3, 4, 5])
    >>> 0.0
    >>> skew([2, 8, 0, 4, 1, 9, 9, 0])
    >>> 0.2650554122698573

    In this example, the input is an array that calculates the skewness 
    using the formula. 

    '''
    n = np.size(data)
    

    average = mean(data) # Calculate the Average
    std_dev = std(data)  # Calculate the Standard Deviation

    # Compute the Skewness according to the formula
    _skewness_ = np.sum((data - average)**3) / (n * std_dev**3) 
   
    return _skewness_


def kurtosis(data):
    # Implement the logic to calculate kurtosis of the data array
    pass
