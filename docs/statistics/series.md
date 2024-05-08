# Moving Average Function

Return the `moving average` of a 1D array. This function implements a cumulative moving average calculation. It's more
efficient than calculating a simple moving average for each element. 

There are two parameters in this function:
1. **data (array_like)**: The 1D array of data for which to calculate the moving average.
2. **window_size (int, optional)**: The size of the window for the moving average. Defaults to 20. Must be less than or equal to the size of the data array.
If the window size is larger than the size of the data array, it raises a ValueError. 

- Examples:

>>>
    >>> from proadv.statistics.series import moving_average
    >>> import numpy as np
    >>> data = np.random.rand(300)
    >>> ma = moving_average(data)

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ma = adv.statistics.series.moving_average(data, window_size=3)
    >>> ma
    array([1. , 1.5, 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ])


# Exponential Moving Average Function

Calculates the `exponential moving average` of a given data series. 
The exponential moving average (EMA) is a type of moving average that places more weight on recent observations while still considering older data. 
It is particularly useful for smoothing noisy data and identifying trends.

There are two parameters in this function:
1. **data (array_like)**: The 1D array of data for which to calculate the exponential moving average.
2. **alpha (float, optional)**: Smoothing factor between 0 and 1. Higher alpha discounts older observations faster. Default is 0.2.

- Example:

>>>
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ema = exponential_moving_average(data, alpha=0.5)
    >>> ema
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])