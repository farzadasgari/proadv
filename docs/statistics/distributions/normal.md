# CDF Function

This function calculates the `cdf` value of an array-like input while checking for NaN values. 
If NaN values are present, it raises a ValueError. It also handles various exceptions that may occur during the operation.
The input data which should be an array or any array-like structure.
If the array contains NaN values, the function will not return a value and will raise a ValueError instead.

- Examples:

>>>
    >>> from proadv.statistics.distributions.normal import cdf
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
    >>> cdf_array = cdf(array)
    >>> cdf_array
    array([0.66780212, 0.72994878, 0.49735891, 0.6718849 , 0.65784697])

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
    >>> adv.statistics.distributions.normal.cdf(array)
    Traceback (most recent call last):
        raise TypeError('array cannot contain NaN values.')
    TypeError: array cannot contain NaN values.