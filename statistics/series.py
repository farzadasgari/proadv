import numpy as np


def moving_average(data, window_size=20):
    """
        Calculates the moving average of a 1D array.

        Parameters
        ------
        data (np.ndarray): The 1D array of data for which to calculate the moving average.
        window_size (int, optional): The size of the window for the moving average.
            Defaults to 20. Must be less than or equal to the size of the data array.

        Returns
        ------
        np.ndarray: The moving average of the data, with the same shape as the input array.

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
