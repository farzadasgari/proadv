# Min Function

Return the minimum of an array input while checking for NaN values. If NaN values are present, it raises a ValueError. 
It also handles various exceptions that may occur during the operation.

- Example 1:

>>>
    >>> import proadv as adv 
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> minimum = adv.statistics.descriptive.min(data)
    >>> print(minimum)
    1

- Example 2:

>>> 
    >>> from proadv.statistics.descriptive import min
    >>> import numpy as np
    >>> data = np.array([1, 2, np.nan, 4, 5])
    >>> minimum = min(data)
    Traceback (most recent call last):
       raise ValueError("The array contains NaN values. The min function cannot be applied to arrays with NaN values.")
    ValueError: The array contains NaN values. The min function cannot be applied to arrays with NaN values.