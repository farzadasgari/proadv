# Skewness Function

Return the sample `skewness` of a data set. For normally distributed data, the skewness should be about zero. For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to zero, statistically speaking.

- Examples:

>>>
    >>> import proadv as adv # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.array([2, 8, 0, 4, 1, 9, 9, 0]) 
    >>> skew = adv.statistics.moment.skewness(data)
    >>> print(skew)
    0.2650554122698573

>>>
    >>> from proadv.statistics.moment import skewness # Option 2: Direct import
    >>> import numpy as np
    >>> data = np.random.rand(20)
    >>> skew = skewness(data) 

>>>
    >>> import proadv as adv # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.arange(1,6)
    >>> skew = adv.statistics.moment.skewness(data)
    >>> skew
    0.0


# Kurtosis Function

Return the `kurtosis` of a dataset. If the input contains NaNs, the function will return NaN. 
Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable. 
This function:
1. converts the input array to a NumPy array 
2. calculates the mean and standard deviation of the data 
3. computes the fourth central moment
4. and then standardizes it to find the kurtosis

- Examples:

>>> from proadv.statistics.moment import kurtosis  # Option 2: Direct import
    >>> import numpy as np
    >>> data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> kurt = kurtosis(data)
    >>> kurt
    2.2

>>>
    >>> import proadv as adv  # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> kurt = adv.statistics.moment.kurtosis(data)
    >>> kurt
    2.2