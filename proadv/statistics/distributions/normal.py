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
    D = array_cdf.shape  # array dimensions
    if array_cdf.size == 0:
        raise ValueError("cannot calculate PDF with empty array.")
    if np.isnan(array_cdf).any():
        raise TypeError('array cannot contain NaN values.')
    if array_cdf.ndim > 1:
        array_cdf = array_cdf.flatten()
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
    array_cdf = array_cdf.reshape(D)
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

        std (float) : The Standard deviation of The normal data.

        mean (float) : The mean of the normal data.

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


def log_pdf(array, std=1, mean=0):
    """
        Calculate the log_pdf value in an array, handling NaN values and exceptions.

        This function calculates the pdf value of an array-like input while checking for NaN values.
            If NaN values are present, it raises a ValueError. It also handles various exceptions that may
            occur during the operation.

        Parameters
        ------
        array (array_like): The input data which should be an array or any array-like structure.

        std (float) : The Standard deviation of The normal data.

        mean (float) : The mean of the normal data.

        Returns
        ------
        Probability density function logarithm (log_pdf): Function to calculate the probability density logarithm of data.
            If the array contains NaN values,
            the function will not return a value
            and will raise a ValueError instead.

        Raises
        ------
        TypeError: If the  element of array is a NaN.
        ValueError: If the array is empty.

        Examples
        ------
        >>> from proadv.statistics.distributions.normal import log_pdf
        >>> import numpy as np
        >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
        >>> logpdf_array = log_pdf(array)
        >>> logpdf_array
        array([-1.0130524 , -1.10661349, -0.91896045, -1.01800619, -1.00159795])

        >>> import proadv as adv
        >>> import numpy as np
        >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
        >>> adv.statistics.distributions.normal.log_pdf(array)
        Traceback (most recent call last):
            raise TypeError('array cannot contain nan values.')
        TypeError: array cannot contain NaN values.

        ------

        """
    array = np.copy(array)
    if array.size == 0:
        raise ValueError("cannot calculate Log_PDF with empty array")
    if np.isnan(array).any():
        raise TypeError('array cannot contain NaN values.')
    logpdf = np.log(pdf(array, std, mean))
    return logpdf


def log_cdf(array):
    """
    Calculate the log_cdf value in an array, handling NaN values and exceptions.

    This function calculates the pdf value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

    Parameters
    ------
    array (array_like): The input data which should be an array or any array-like structure.

    Returns
    ------
    Cumulative distribution function logarithm (log_cdf): Function to calculate the Cumulative distribution logarithm of data.
        If the array contains NaN values,
        the function will not return a value
        and will raise a ValueError instead.

    Raises
    ------
    TypeError: If the  element of array is a NaN.
    ValueError: If the array is empty.

    Examples
    ------
    >>> from proadv.statistics.distributions.normal import log_pdf
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
    >>> logcdf_array = log_cdf(array)
    >>> logcdf_array
    array([-0.40376338, -0.31478092, -0.69844337, -0.39766824, -0.41878294])

    >>> import proadv as adv
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
    >>> adv.statistics.distributions.normal.log_cdf(array)
    Traceback (most recent call last):
        raise TypeError('array cannot contain nan values.')
    TypeError: array cannot contain NaN values.

    ------

    """
    array_cdf = np.copy(array)
    if array_cdf.size == 0:
        raise ValueError("cannot calculate Log_CDF with empty array.")
    if np.isnan(array_cdf).any():
        raise TypeError('array cannot contain NaN values.')
    logcdf = np.log(cdf(array))
    return logcdf


def sf(array):
    """
    Calculate the sf value in an array, handling NaN values and exceptions.

    This function calculates the sf value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

    Parameters
    ------
    array (array_like): The input data which should be an array or any array-like structure.

    Returns
    ------
    Survival function (sf): Function to calculate the Survival of data.
        If the array contains NaN values,
        the function will not return a value
        and will raise a ValueError instead.

    Raises
    ------
    TypeError: If the  element of array is a NaN.
    ValueError: If the array is empty.

    Examples
    ------
    >>> from proadv.statistics.distributions.normal import sf
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
    >>> sf_array = sf(array)
    >>> sf_array
    array([0.33219788, 0.27005122, 0.50264109, 0.3281151 , 0.34215303])

    >>> import proadv as adv
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
    >>> adv.statistics.distributions.normal.sf(array)
    Traceback (most recent call last):
        raise TypeError('array cannot contain nan values.')
    TypeError: array cannot contain NaN values.

    ------

    """
    array_sf = np.copy(array)
    if array_sf.size == 0:
        raise ValueError("cannot calculate SF with empty array.")
    if np.isnan(array_sf).any():
        raise TypeError('array cannot contain NaN values.')
    sf_array = 1 - cdf(array)
    return sf_array


def log_sf(array):
    """
    Calculate the log_sf value in an array, handling NaN values and exceptions.

    This function calculates the log_sf value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

    Parameters
    ------
    array (array_like): The input data which should be an array or any array-like structure.

    Returns
    ------
    Survival function logarithm (log_sf): Function to calculate the Survival of data.
        If the array contains NaN values,
        the function will not return a value
        and will raise a ValueError instead.

    Raises
    ------
    TypeError: If the  element of array is a NaN.
    ValueError: If the array is empty.

    Examples
    ------
    >>> from proadv.statistics.distributions.normal import log_sf
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029,  0.44512392,  0.4065942 ])
    >>> logsf_array = log_sf(array)
    >>> logsf_array
    array([-1.10202446, -1.30914362, -0.68787889, -1.11439081, -1.07249719])

    >>> import proadv as adv
    >>> import numpy as np
    >>> array = np.array([0.43385221, 0.61265808, -0.00662029, np.nan,  0.44512392,  0.4065942])
    >>> adv.statistics.distributions.normal.log_sf(array)
    Traceback (most recent call last):
        raise TypeError('array cannot contain nan values.')
    TypeError: array cannot contain NaN values.

    ------

    """
    logsf = np.log(sf(array))
    return logsf
