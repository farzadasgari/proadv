# Moving Average Function

Return the `moving average` of a 1D array. This function implements a cumulative moving average calculation. It's more
efficient than calculating a simple moving average for each element. 
There are two arguments in this function:
1. data (array_like): The 1D array of data for which to calculate the moving average.
2. window_size (int, optional): The size of the window for the moving average. Defaults to 20. Must be less than or equal to the size of the data array.

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
