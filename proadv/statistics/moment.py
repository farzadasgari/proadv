import numpy as np
from .descriptive import mean
from .spread import std


def skewness(data):
    """
    Compute the sample skewness of a data set. 

    Parameters
    ------
    data (array_like): The 1D array of data for which to calculate the Skewness.

    Returns
    ------
    skew: The skewness of the data

    References
    ------
    [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section 2.2.24.1

    Notes
    ------
    For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.

    Examples
    ------
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([2, 8, 0, 4, 1, 9, 9, 0]) 
    >>> skew = adv.statistics.moment.skewness(data)
    >>> skew
    0.2650554122698573

    ------

    >>> from proadv.statistics.moment import skewness
    >>> import numpy as np
    >>> data = np.random.rand(20)
    >>> skew = skewness(data) 

    ------

    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.arange(1,6)
    >>> skew = adv.statistics.moment.skewness(data)
    >>> skew
    0.0
    """
    n = np.size(data)

    average = mean(data)  # Calculate the Average
    std_dev = std(data)  # Calculate the Standard Deviation

    # Compute the Skewness according to the formula
    skew = np.sum((data - average) ** 3) / (n * std_dev ** 3)

    return skew


def kurtosis(x):
    """
    Compute the kurtosis of a dataset.

    Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
        This function calculates kurtosis using the Pearson method, which is the standardized fourth central moment.

    Parameters
    ------
    x (array_like): An array containing the data points. The array will be flattened if it is not already 1-D.

    Returns
    ------
    standardized_moment (float): The kurtosis of the dataset. If the input contains NaNs, the function will return NaN.

    Notes
    ------
    This function converts the input array to a NumPy array, calculates the mean and standard deviation of the data,
    computes the fourth central moment, and then standardizes it to find the kurtosis.

    Examples
    ------
    >>> from proadv.statistics.moment import kurtosis
    >>> import numpy as np
    >>> array = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> kurt = kurtosis(array)
    >>> kurt
    2.2

    ------
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    >>> kurt = adv.statistics.moment.kurtosis(data)
    >>> kurt
    2.2
    """

    # Convert data to a numpy array
    x = np.array(x)

    # Calculate the mean
    average = mean(x)

    # Calculate the standard deviation
    standard_dev = std(x)

    # Calculate the fourth central moment
    fourth_moment = mean((x - average) ** 4)

    # Standardize the fourth moment
    standardized_moment = fourth_moment / standard_dev ** 4

    return standardized_moment  # Returns the standardized moment which basically is kurtosis
