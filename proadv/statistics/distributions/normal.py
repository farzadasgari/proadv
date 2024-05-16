import numpy as np
from proadv.statistics.moment import kurtosis, skewness


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


def skewtest(data, alternative='two-sided'):
    """
    Perform the skewness test for normality.

    The skewness test measures whether the skewness of the provided dataset
    differs significantly from that of a normally distributed dataset.

    Parameters
    ------
    data : array_like
        The data to be tested.
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis to test. Default is 'two-sided'.

    Returns
    ------
    statistic : float or ndarray
        The computed z-score for this test.
    pvalue : float or ndarray
        The p-value for the hypothesis test.

    Raises
    ------
    ValueError
        If the number of samples is less than 8.

    Notes
    ------
    The null hypothesis for this test is that the skewness of the population
    that the sample was drawn from is the same as that of a corresponding
    normal distribution.
    """
    # Convert input data to a numpy array
    data = np.asarray(data)

    # Calculate the skewness of the data
    skew = skewness(data)

    # Calculate the number of samples
    sample_size = data.shape[0]

    # Check if the sample size is valid
    if sample_size < 8:
        raise ValueError("skewtest requires at least 8 samples; {} samples were given.".format(sample_size))

    # Calculate the adjustment factor for the skewness
    adjustment_factor = np.sqrt(((sample_size + 1) * (sample_size + 3)) / (6.0 * (sample_size - 2)))

    # Calculate the adjusted skewness
    adjusted_skewness = skew * adjustment_factor

    # Calculate the beta2 constant
    beta2 = (3.0 * (sample_size ** 2 + 27 * sample_size - 70) * (sample_size + 1) * (sample_size + 3)) / (
            (sample_size - 2.0) * (sample_size + 5) * (sample_size + 7) * (sample_size + 9))

    # Calculate the w2 constant
    w2 = -1 + np.sqrt(2 * (beta2 - 1))

    # Calculate the delta constant
    delta = 1 / np.sqrt(0.5 * np.log(w2))

    # Calculate the alpha constant
    alpha = np.sqrt(2.0 / (w2 - 1))

    # Ensure adjusted skewness is not zero for calculation
    adjusted_skewness = np.where(adjusted_skewness == 0, 1, adjusted_skewness)

    # Calculate the z score
    z = delta * np.log(adjusted_skewness / alpha + np.sqrt((adjusted_skewness / alpha) ** 2 + 1))

    # Initialize p_value to None to ensure it has a value in all code paths
    p_value = None

    # Calculate the p-value based on the z score
    if alternative == 'two-sided':
        p_value = 2 * (1 - cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - cdf(z)
    elif alternative == 'less':
        p_value = cdf(z)

    # Check if p_value was set, raise an error if not
    if p_value is None:
        raise ValueError("Invalid value for 'alternative'. Please choose from 'two-sided', 'less', or 'greater'.")

    return z, p_value


def kurtotest(data, alternative='two-sided'):
    """
    Perform the kurtosis test for normality.

    This function tests the null hypothesis that the kurtosis of the data
    set is the same as that of a normal distribution: a kurtosis of three.

    Parameters
    ------
    data : array_like
        Array of sample data.
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis to test. Default is 'two-sided'.

    Returns
    ------
    statistic : float or ndarray
        The computed z-score for this test.
    pvalue : float or ndarray
        The p-value for the hypothesis test.

    Raises
    ------
    ValueError
        If the number of observations is less than 5.

    Notes
    ------
    The null hypothesis for the kurtosis test is that the kurtosis of the
    population from which the sample was drawn is that of the normal
    distribution: kurtosis = 3(n-1)/(n+1).

    """
    # Convert input data to a numpy array
    data = np.asarray(data)

    # Calculate the number of observations
    num_observations = data.shape[0]

    # Validate the number of observations
    if num_observations < 5:
        raise ValueError(
            "kurtosistest requires at least 5 observations; {} observations were given.".format(num_observations))

    # Calculate the kurtosis of the data
    kurtosis_value = kurtosis(data)

    # Expected kurtosis for a normal distribution
    expected_kurtosis = 3.0 * (num_observations - 1) / (num_observations + 1)

    # Variance of the kurtosis estimate
    variance_kurtosis = 24.0 * num_observations * (num_observations - 2) * (num_observations - 3) / (
            (num_observations + 1) * (num_observations + 1) * (num_observations + 3) * (num_observations + 5))

    # Standardized test statistic
    z_score = (kurtosis_value - expected_kurtosis) / np.ma.sqrt(variance_kurtosis)

    # Adjustments for small sample sizes
    sqrt_beta1 = 6.0 * (num_observations ** 2 - 5 * num_observations + 2) / (
            (num_observations + 7) * (num_observations + 9)) * np.ma.sqrt(
        (6.0 * (num_observations + 3) * (num_observations + 5)) / (
                num_observations * (num_observations - 2) * (num_observations - 3)))
    a = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + np.ma.sqrt(1 + 4.0 / (sqrt_beta1 ** 2)))

    # Terms for the calculation of the z-score
    term1 = 1 - 2.0 / (9.0 * a)
    denom = 1 + z_score * np.ma.sqrt(2 / (a - 4.0))

    # Ensure denom is an array before assignment
    if isinstance(denom, np.ma.MaskedArray):
        denom[denom == 0.0] = np.ma.masked
    elif np.isscalar(denom) and denom == 0.0:
        denom = np.ma.masked

    term2 = np.ma.power((1 - 2.0 / a) / denom, 1 / 3.0)

    # Final z-score calculation
    z_score_final = (term1 - term2) / np.ma.sqrt(2 / (9.0 * a))

    # Initialize p_value to None to ensure it has a value in all code paths
    p_value = None

    # Calculate the p-value based on the z-score
    if alternative == 'two-sided':
        p_value = 2 * (1 - cdf(abs(z_score_final)))
    elif alternative == 'greater':
        p_value = 1 - cdf(z_score_final)
    elif alternative == 'less':
        p_value = cdf(z_score_final)

    # Check if p_value was set, raise an error if not
    if p_value is None:
        raise ValueError("Invalid value for 'alternative'. Please choose from 'two-sided', 'less', or 'greater'.")

    return z_score_final, p_value


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
