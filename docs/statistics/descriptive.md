# Min Function

Return the `minimum` of an array input while checking for NaN values. If NaN values are present, it raises a ValueError. 
It also handles various exceptions that may occur during the operation.

- Examples:

>>>
    >>> import proadv as adv 
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> minimum = adv.statistics.descriptive.min(data)
    >>> print(minimum)
    1

>>> 
    >>> from proadv.statistics.descriptive import min
    >>> import numpy as np
    >>> data = np.array([1, 2, np.nan, 4, 5])
    >>> minimum = min(data)
    Traceback (most recent call last):
       raise ValueError("The array contains NaN values. The min function cannot be applied to arrays with NaN values.")
    ValueError: The array contains NaN values. The min function cannot be applied to arrays with NaN values.

>>>
    >>> from proadv.statistics.descriptive import min 
    >>> import numpy as np
    >>> data = np.random.rand(20) 
    >>> minimum = min(data)


# Max Function

Return the `maximum` of an array input while checking for NaN values. If NaN values are present, it raises a ValueError. 
It also handles various exceptions that may occur during the operation.

- Examples:

>>>
    >>> from proadv.statistics.descriptive import max
    >>> import numpy as np
    >>> data = np.array([1, 2, np.nan, 4, 5])
    >>> maximum = max(data)
    Traceback (most recent call last):
        raise ValueError("The array contains NaN values. The max function cannot be applied to arrays with NaN values.")
    ValueError: The array contains NaN values. The max function cannot be applied to arrays with NaN values.

>>>
    >>> import proadv as adv 
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> maximum = adv.statistics.descriptive.max(data)
    >>> print(maximum)
    5

>>>
    >>> from proadv.statistics.descriptive import max 
    >>> import numpy as np
    >>> data = np.arange(2,10)
    >>> maximum = max(data)
    >>> print(maximum)
    9


# Mean Function

Return the `mean` of a dataset, handling non-numeric and numeric data. Dataset can be a single number, a list of numbers, or a list containing both numbers and strings. Non-numeric strings are converted to floats if possible, and ignored if not. 

- Examples:

>>>
    >>> import proadv as adv 
    >>> import numpy as np
    >>> array = np.array([1, 2, 3])
    >>> mean_array = adv.statistics.descriptive.mean(array)
    >>> print(mean_array)
    2.0

>>>
    >>> from proadv.statistics.descriptive import mean 
    >>> import numpy as np
    >>> array = np.array([1, 2, 3, 4, 5])
    >>> mean_array = mean(array)
    >>> print(mean_array)
    3.0

>>>
    >>> from proadv.statistics.descriptive import mean 
    >>> import numpy as np
    >>> array = np.array([1, '2', 3.5, 'not a number'])
    >>> mean_array = mean(array)
    >>> print(mean_array)
    2.1666666666666665

>>>
    >>> import proadv as adv 
    >>> import numpy as np
    >>> array = np.array([])
    >>> mean_array = adv.statistics.descriptive.mean(array)
    >>> print(mean_array)
    Invalid input.


# Median Function

Returnn the `median` along the specified axis. If the element of array is a string, it raises a TypeError. 
Also, If the array is empty, it raises a ValueError.

- Examples:

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([14, 8, 11, 10, 5, 7])
    >>> med = adv.statistics.descriptive.median(data)
    >>> print(med)
    9.0
    
>>>
    >>> from proadv.statistics.descriptive import median
    >>> import numpy as np
    >>> data = np.random.rand(15)
    >>> med = median(data) 

>>>
    >>> import proadv as adv
    >>> import numpy as np 
    >>> data = np.arange(14,45)
    >>> med = adv.statistics.descriptive.median(data)
    >>> print(med)
    29.0


# Mod Function

This function computes an array of the modal (most common) value in the passed array. 
It returns:
1. **mode_value (int)**: The mode of the data
2. **frequency (int)**: The number of repetitions of the mode

- Exampls:

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([3, 0, 3, 7, 2])
    >>> values, counts = adv.statistics.descriptive.mode(data)
    >>> values
    3
    >>> counts
    2

>>>
    >>> from proadv.statistics.descriptive import mode
    >>> import numpy as np
    >>> data = np.array([4, 6, 12, 4, 15, 4, 6, 16])
    >>> values, counts = mode(data)
    >>> values
    4
    >>> counts
    3