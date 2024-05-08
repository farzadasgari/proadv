import numpy as np


def cdf(array):
    """
    Calculate the cdf value in an array, handling NaN values and exceptions.

    This function calculates the cdf value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

    Parameters
    ------
    array (array_like): The input data which should be an array or any array-like structure.

    Returns
    ------
    Cumulative distribution function (cdf): Function to calculate the cumulative distribution of data.
        If the array contains NaN values,
        the function will not return a value
        and will raise a ValueError instead.

    Raises
    ------
    TypeError: If the  element of array is a NaN.
    ValueError: If the array is empty.

    Examples
    ------
    >>> from proadv.statistics.distributions.normal import cdf
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
    >>> cdf_array = cdf(array)
    >>> cdf_array
    array([0.66780212, 0.72994878, 0.49735891, 0.6718849 , 0.65784697])


    >>> import proadv as adv
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
    >>> adv.statistics.distributions.normal.cdf(array)
    Traceback (most recent call last):
        raise TypeError('array cannot contain NaN values.')
    TypeError: array cannot contain NaN values.

    """

    from math import erf, erfc
    array_cdf = np.copy(array)
    if array_cdf.size == 0:
        raise ValueError("cannot calculate PDF with empty array.")
    if np.isnan(array_cdf).any():
        raise TypeError('array cannot contain NaN values.')
    np_sqrt = 1.0 / np.sqrt(2)
    array_ns = array_cdf * np_sqrt
    absolute_value = np.fabs(array_ns)
    j = 0
    for i in absolute_value:
        if i < np_sqrt:
            array_cdf[j] = 0.5 + 0.5 * erf(array_ns[j])
        else:
            y = 0.5 * erfc(i)
            if array_ns[j] > 0:
                array_cdf[j] = 1.0 - y
            else:
                array_cdf[j] = y
        j += 1

    return array_cdf


def pdf(array, std=1, mean=0):
    """
        Calculate the pdf value in an array, handling NaN values and exceptions.

        This function calculates the pdf value of an array-like input while checking for NaN values.
            If NaN values are present, it raises a ValueError. It also handles various exceptions that may
            occur during the operation.

        Parameters
        ------
        array (array_like): The input data which should be an array or any array-like structure.

        Returns
        ------
        Probability density function (pdf): Function to calculate the probability density of data.
            If the array contains NaN values,
            the function will not return a value
            and will raise a ValueError instead.

        Raises
        ------
        TypeError: If the  element of array is a NaN.
        ValueError: If the array is empty.

        Examples
        ------
        >>> from proadv.statistics.distributions.normal import pdf
        >>> import numpy as np
        >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
        >>> pdf_array = pdf(array)
        >>> pdf_array
        array([0.36310893, 0.33067691, 0.39893354, 0.36131462, 0.36729206])

        >>> import proadv as adv
        >>> import numpy as np
        >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
        >>> adv.statistics.distributions.normal.pdf(array)
        Traceback (most recent call last):
            raise TypeError('array cannot contain nan values.')
        TypeError: array cannot contain NaN values.

        ------

        """
    array = np.copy(array)
    if array.size == 0:
        raise ValueError("cannot calculate PDF with empty array")
    if np.isnan(array).any():
        raise TypeError('array cannot contain NaN values.')
    x = (-0.5 * np.log(2 * np.pi)) - np.log(std)
    y = np.power(array - mean, 2) / (2 * (std * std))
    array_pdf = np.exp(x - y)
    return array_pdf


