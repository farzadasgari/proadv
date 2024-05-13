import numpy as np
import matplotlib.pyplot as plt


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


def theoretical_quantiles(x):
    """
    Calculate the theoretical quantiles for a probability plot.

    Parameters
    ------
    x : array_like
        The data for which to calculate the theoretical quantiles.

    Returns
    ------
    quantiles : ndarray
        Theoretical quantiles calculated based on Filliben's estimate.
    """
    n = len(x)
    # Calculate the theoretical quantiles using Filliben's formula
    return np.array([(i - 0.3175) / (n + 0.365) if i != 1 and i != n else
                     0.5 ** (1 / n) if i == n else
                     1 - 0.5 ** (1 / n) for i in range(1, n + 1)])


def probplot(x, dist='norm', sparams=(), fit=True, plot=None, rvalue=False):
    """
    probplot function creates a probability plot, which is a graphical technique for
    assessing whether or not a data set follows a given distribution, such as the normal distribution.
    The main steps involved in a probplot are:

    1.Sorting Data: The sample data is sorted in ascending order.
    2.Calculating Theoretical Quantiles: Theoretical quantiles are calculated from
    the specified distribution (normal by default).
    3.Fitting: If requested,
    a least-squares regression (best-fit line) is performed to see how well the data fits the theoretical distribution.
    4.Plotting: The sorted data is plotted against the theoretical quantiles to visualize the distribution of the data.
    5.Assessment: If the data points closely follow the fitted line,
    the data is likely to follow the specified distribution.

    Parameters
    ------
    x : array_like
        Sample/response data from which probplot creates the plot.
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name. The default is 'norm'.
    sparams : tuple, optional
        Distribution-specific shape parameters.refers to additional parameters specific to the distribution being used
        in the probability plot. For example, if you’re working with a distribution that requires a shape parameter,
        like a t-distribution which requires degrees of freedom, you would pass that in as sparams.
        It allows the function to adjust the theoretical distribution based on these parameters.
    fit : bool, optional
        Fit a least-squares regression line to the sample data if True.
    plot : object, optional
        If given, plots the quantiles and the least squares fit if fit is True.
    rvalue : bool, optional
         is an optional boolean parameter that, when set to True,
        includes the coefficient of determination (denoted as ( R^2 )) on the plot.
        The coefficient of determination is a statistical measure that
         explains how well the data fits a statistical model – in this case, the best-fit line on the probability plot.
        A higher ( R^2 ) value indicates a better fit of the data to the line.

    Returns
    ------
    (osm, osr) : tuple of ndarrays
        Tuple of theoretical quantiles (osm) and ordered responses (osr).
    (slope, intercept, r) : tuple of floats, optional
        Tuple containing the result of the least-squares fit.

    Examples
    ------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    # Generate some data that follows a normal distribution
    >>> data = np.random.normal(loc=0, scale=1, size=100)
    # Create a subplot to display the plot
    >>> fig, ax = plt.subplots()
    # Call the probplot function with the generated data and the subplot
    >>> res = probplot(data, plot=ax)
     # Show the plot with the probability plot and the best-fit line
    >>> plt.show()
    """
    # Sort the data and calculate the theoretical quantiles
    data = np.sort(x)
    # osm stands for Order Statistic Medians.
    # These are the theoretical quantiles calculated from the specified distribution.
    # They represent the position of each data point if the data were to follow the theoretical distribution perfectly.
    osm = theoretical_quantiles(data)
    # Initialize slope and intercept to None
    slope = None
    intercept = None
    # Check if a distribution object is provided and has a ppf method
    theoretical = dist.ppf(osm, *sparams) if hasattr(dist, 'ppf') else np.sort(np.random.normal(size=len(data)))

    # Initialize r_squared to None or an appropriate default value
    r_squared = None

    if fit:
        # Perform a least-squares regression to fit a line to the data
        # np.polyfit returns a list of coefficients,
        # where the first element is the slope and the second is the intercept
        coefficients = np.polyfit(theoretical, data, 1)
        # Assign the first element to slope and the second to intercept
        slope, intercept = coefficients[0], coefficients[1]
        # osr stands for Ordered Sample Responses.
        # These are the actual data points from your sample, sorted in ascending order.
        # They are plotted against the OSM to assess how well the sample data fits the theoretical distribution.
        osr = slope * theoretical + intercept

        # Calculate R^2, the coefficient of determination
        ss_res = np.sum((data - osr) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    else:
        osr = data

    if plot is not None:
        plot.plot(theoretical, data, 'o')
        if fit:
            plot.plot(theoretical, osr, 'r-')
            if rvalue and r_squared is not None:
                # Use r_squared in the plot text only if it's been calculated
                plot.text(0.8, 0.9, f'$R^2 = {r_squared:.3f}$', horizontalalignment='center',
                          verticalalignment='center', transform=plot.transAxes)

    # Ensure that r_squared is returned only if it's been calculated
    if fit:
        # Return the theoretical quantiles and ordered responses, along with the fit results
        return (osm, osr), (slope, intercept, r_squared)
    else:
        return osm, osr

