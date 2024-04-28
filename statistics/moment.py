import numpy as np
import desc
from desc import mean
from spread import std

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


def kurtosis(x):
    """
    Compute the kurtosis of a dataset.

 Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
    This function calculates kurtosis using the Pearson method, which is the standardized fourth central moment.

    Parameters
    ----------
    x : array_like
        An array containing the data points. The array will be flattened if it is not already 1-D.

    Returns
    -------
    float
        The kurtosis of the dataset. If the input contains NaNs, the function will return NaN.

    Notes
    -----
    This function converts the input array to a NumPy array, calculates the mean and standard deviation of the data,
    computes the fourth central moment, and then standardizes it to find the kurtosis.

    Examples
    --------
    >>> data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    >>> kurtosis(data)
    out: 2.2
    """

    # Convert data to a numpy array
    x = np.array(x)

    # Calculate the mean
    average = desc.mean(x)

    # Calculate the standard deviation
    standard_dev = desc.standard_deviation(x)

    # Calculate the fourth central moment
    fourth_moment = desc.mean((x - average) ** 4)

    # Standardize the fourth moment
    standardized_moment = fourth_moment / standard_dev ** 4

    return standardized_moment  # Returns the standardized moment which basically is kurtosis
