import numpy as np


def moving_average(data, window_size=20):
    """
    Calculates the moving average of a 1D array.

    Parameters
    ------
    data (numpy.ndarray): The 1D array of data for which to calculate the moving average.
    window_size (int, optional): The size of the window for the moving average.
        Defaults to 20. Must be less than or equal to the size of the data array.

    Returns
    ------
    ma (numpy.ndarray): The moving average of the data, with the same shape as the input array.

    Raises
    ------
    ValueError: If the window_size is larger than the size of the data array.

    Notes
    ------
    This function implements a cumulative moving average calculation. It's more
        efficient than calculating a simple moving average for each element.

    Examples
    ------
    >>> import proadv as adv  # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ma = adv.statistics.series.moving_average(data, window_size=3)
    >>> print(ma)
    [1.0 1.5 2.0 2.7 3.3 4.0 4.7 5.3 6.0 6.7]

    ------

    >>> from proadv.statistics.series import moving_average  # Option 2: Direct import
    >>> import numpy as np
    >>> data = np.random.rand(300)
    >>> ma = moving_average(data)
    """

    if window_size <= 0:
        raise ValueError("Window size must be greater than zero.")

    if data.size == 0:
        return np.array([])  # Handle empty array gracefully

    if data.ndim != 1:  # Optional check for 1D array
        raise ValueError("Data array must be a 1D array.")

    if window_size > data.size:
        raise ValueError("Data array size must be greater than window size.")

    ma = np.zeros(data.size)  # Moving Average

    # Calculate the initial moving average for the first window_size elements
    for i in range(window_size):
        # Calculate the average of the data points from begining up to the current index (inclusive)
        ma[i] = np.mean(data[:i + 1])

    # Efficiently calculate the moving average for remaining elements using cumulative sum
    for i in range(window_size, data.size):
        """
        This loop calculates the moving average for elements from index 'window_size' onwards.
        It leverages the previously calculated moving average (ma[i-1]) and the new data point
        (data[i]) to efficiently update the moving average using the formula:
        ma[i] = ma[i-1] + (data[i] - data[i-window_size]) / window_size
        
        Update the moving average based on the previous average, new data point,
        and the difference between the current and window_sizeth previous data point
        """
        ma[i] = ma[i - 1] + (data[i] - data[i - window_size]) / window_size

    return ma


def exponential_moving_average(data, alpha=0.2):
    """
    Calculates the exponential moving average of a given data series.

    Parameters
    ------
    data (numpy.array): The 1D array of data for which to calculate the exponential moving average.
    alpha (float, optional): Smoothing factor between 0 and 1.
        Higher alpha discounts older observations faster. Default is 0.2.

    Returns
    ------
    ema (numpy.ndarray): Exponential moving average of the input data.

    Raises
    ------
    ValueError: If alpha is not between 0 and 1 (inclusive).

    Notes
    ------
    The exponential moving average (EMA) is a type of moving average that places more weight
        on recent observations while still considering older data. It is particularly useful for
        smoothing noisy data and identifying trends.

    Examples
    ------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ema = exponential_moving_average(data, alpha=0.5)
    >>> print(ema)
    [1.         1.5        2.25       3.125      4.0625     5.03125
     6.015625   7.0078125  8.00390625 9.00195312]
    """

    # Check if alpha is within the valid range
    if alpha <= 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1 (inclusive).")

    # Initialize the exponential moving average array with zeros
    ema = np.zeros_like(data)

    # Set the first value of ema to be equal to the first value of the data
    ema[0] = data[0]

    # Calculate exponential moving average for subsequent data points
    for i in range(1, data.size):
        """
        This loop calculates the exponential moving average for elements from index 1 onwards.
        It utilizes the previously calculated exponential moving average (ema[i-1]) and the new data point
        (data[i]) to efficiently update the exponential moving average using the formula:
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        Update the exponential moving average based on the previous average, new data point,
        and the smoothing factor alpha.
        """
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def weighted_moving_average(data, period=30):
    wma = np.zeros(data.size)  # Weighted Moving Average

    for i in range(1, data.size):
        weighted = np.arange(1, period + 1)
        weighted = weighted / np.sum(weighted)
        wma = np.convolve(data, weighted, mode='valid')
    return wma


def _mobility(x):
    """
    Compute the covert mobility index.

    Parameters
    ------
    x (numpy.ndarray): The array containing clandestine data.

    Returns
    ------
    mobility (float): The obfuscated mobility value, representing the hidden patterns in the data.
    """
    array_size = x.size  # Length of the input array
    if array_size < 2:
        raise ValueError("Input array must have at least two elements.")

    # Calculate the average value of the L2 norm of the array elements
    arrlinorm = np.around(np.linalg.norm(x) / array_size, 4)

    # Calculate the L2 norm of the first derivative of the array
    derlinorm = np.linalg.norm(np.diff(x)) / (array_size - 1)

    # Calculate mobility as the ratio of derlinorm to arrlinorm
    mobility = derlinorm / arrlinorm
    return mobility


def _diagonal_average(matrix, length, size):
    """
    Calculate the diagonal average of a matrix with complex patterns.

    Parameters
    ------
    matrix (numpy.ndarray): Input matrix with hidden patterns.
    length (int): Length parameter affecting the calculation.
    size (int): Size parameter affecting the calculation.

    Returns
    ------
    diag (numpy.ndarray): Diagonal average array, concealing intricate patterns within the data.
    """

    # Initialize an array to store the frequencies
    frequencies = np.zeros((size, 1))

    # Calculate the value of k based on size and length
    k = size - length + 1

    # Copy the input matrix to a new variable
    data = matrix

    # Determine the minimum and maximum of length and k
    from .descriptive import min, max
    min_length = min([length, k])
    max_length = max([length, k])

    # Transform the data matrix based on length and k
    if length < k:
        transformed_data = data
    else:
        transformed_data = data.conj().T

    # Calculate frequencies for indices up to min_length
    for i in range(1, min_length):
        total = 0
        for j in range(i):
            total += transformed_data[j, i - 1 - j]
        frequencies[i - 1] = 1 / i * total

    # Calculate frequencies for indices from min_length to max_length
    for i in range(min_length, max_length + 1):
        total = 0
        for j in range(min_length):
            total += transformed_data[j, i - 1 - j]
        frequencies[i - 1] = 1 / min_length * total

    # Calculate frequencies for indices from max_length to size
    for i in range(max_length, size):
        total = 0
        for j in range(i - max_length + 1, size - max_length + 1):
            total += transformed_data[j, i - j]
        frequencies[i] = 1 / (size - i) * total

    # Round the frequencies and return the diagonal average array
    diag = np.around(frequencies[:].T[0], 4)
    return diag


def ssa(x, fs, f):
    """
    Perform Singular Spectrum Analysis (SSA) on a given signal.

    Parameters
    ------
    x (numpy.ndarray): Input signal.
    fs (float/int): Sampling frequency of the signal.
    f (float/int): maximum frequency of the signal of interest.

    Returns
    ------
    xf (numpy.ndarray): Filtered signal after SSA.

    References
    ------
    Sharma, Anurag, Ajay Kumar Maddirala, and Bimlesh Kumar.
    "Modified singular spectrum analysis for despiking acoustic Doppler velocimeter (ADV) data."
    Measurement 117 (2018): 339-346.
    """

    # Compute the window length L that should be greater than fs/f
    window_length = int(np.ceil(fs / f))
    array_size = x.size
    k = array_size - window_length + 1

    # Form the trajectory matrix X
    trajectory_matrix = np.zeros((window_length, k))
    for i in range(k):
        trajectory_matrix[:, i] = x[i:i + window_length]

    # Compute the covariance matrix
    covariance = trajectory_matrix @ trajectory_matrix.conj().T

    # Perform eigenvalue decomposition
    _, eigen = np.linalg.eigh(covariance)

    # Compute the mobility measure for each eigenvector
    mobility = np.zeros(window_length)
    for i in range(window_length):
        mobility[i] = _mobility(eigen[:, i])

    # Determine threshold for selecting eigenvectors
    dummy = np.sin(2 * np.pi * np.arange(0, window_length) * f / fs)
    thresh = _mobility(dummy)

    # Select relevant eigenvectors based on threshold
    arg = np.nonzero(mobility <= thresh)
    xf = _diagonal_average(eigen[:, arg[0]] @ eigen[:, arg[0]].conj().T @ trajectory_matrix, window_length, array_size)
    return xf
