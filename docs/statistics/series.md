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
If alpha is not between 0 and 1 (inclusive), It raises a ValueError. 

- Example:

>>>
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ema = exponential_moving_average(data, alpha=0.5)
    >>> ema
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# Weighted Moving Average Function

Calculates the `weighted moving average` of a 1D array. 
The weighted moving average (WMA) is a type of moving average that assigns a greater weighting to the most recent data points, and less weighting to data points in the distant past.

There are two parameters in this function:
1. **data (array_like)**: The 1D array of data for which to calculate the weighted moving average.
2. **period (int, optional)**: The period for the weighted moving average. Defaults to 20. Must be less than or equal to the size of the data array.
If the period is larger than the size of the data array, it raises a ValueError. 

- Examples:

>>>
    >>> import proadv as adv  
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> wma = adv.statistics.series.weighted_moving_average(data, period=3)
    >>> wma
    array([2.33333333, 3.33333333, 4.33333333, 5.33333333, 6.33333333,
           7.33333333, 8.33333333])

>>>
    >>> from proadv.statistics.series import weighted_moving_average  
    >>> import numpy as np
    >>> data = np.random.rand(300)
    >>> wma = weighted_moving_average(data)


# _mobility Function

Return the covert `mobility` index.
In this function, The obfuscated mobility value, representing the hidden patterns in the data.


# _diagonal_average Function

Calculate the `diagonal average` of a matrix with complex patterns.

There are three parameters in this function:
1. **matrix (array_like)**: Input matrix with hidden patterns.
2. **length (int)**: Length parameter affecting the calculation.
3. **size (int)**: Size parameter affecting the calculation.


# SSA Function

Perform `Singular Spectrum Analysis (SSA)` on a given signal. It returns filtered signal after SSA.

There are three parameters in this function:
1. **x (array_like)**: Input signal.
2. **fs (float/int)**: Sampling frequency of the signal.
3. **f (float/int)**: maximum frequency of the signal of interest.