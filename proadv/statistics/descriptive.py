import numpy as np


def min(data):
    """
    Calculate the minimum value in an array, handling NaN values and exceptions.

    This function calculates the minimum value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

    Parameters
    ------
    data (array_like): The input data which should be an array or any array-like structure.

    Returns
    ------
    minimum (numerical): The minimum value in the data. If the array contains NaN values, the function will not return a value
        and will raise a ValueError instead.

    Raises
    ------
    TypeError: If the  element of array is a string.
    ValueError: If the array is empty.

    Examples
    ------
    >>> import proadv as adv 
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> minimum = adv.statistics.descriptive.min(data)
    >>> minimum
    1
    
    ------

    >>> from proadv.statistics.descriptive import min
    >>> import numpy as np
    >>> data = np.array([1, 2, np.nan, 4, 5])
    >>> minimum = min(data)
    Traceback (most recent call last):
       raise ValueError("The array contains NaN values. The min function cannot be applied to arrays with NaN values.")
    ValueError: The array contains NaN values. The min function cannot be applied to arrays with NaN values.

    ------

    >>> from proadv.statistics.descriptive import min 
    >>> import numpy as np
    >>> data = np.random.rand(20)
    >>> minimum = min(data)
    """

    for i in data:
        if isinstance(i,
                      str):  # isinstance returns True if the specified object is of the specified type, otherwise False
            raise TypeError("String cannot be placed as an element of an array")
    if np.isnan(data).any():
        raise ValueError("The array contains NaN values. The min function cannot be applied to arrays with NaN values.")
    if np.size(data) == 0:  # The array cannot be empty
        raise ValueError("cannot calculate minimum with empty array")

    minimum = np.min(data)  # Calculate the minimum
    return minimum


def max(data):
    """
    Calculate the maximum value in an array, handling NaN values and exceptions.

    This function calculates the maximum value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

    Parameters
    ------
    data (array_like): The input data which should be an array or any array-like structure.

    Returns
    ------
    maximum :The maximum value in the data. If the array contains NaN values, the function will not return a value
        and will raise a ValueError instead.

    Raises
    ------
    TypeError: If the  element of array is a string.
    ValueError: If the array is empty.

    Examples
    ------
    >>> from proadv.statistics.descriptive import max
    >>> import numpy as np
    >>> data = np.array([1, 2, np.nan, 4, 5])
    >>> maximum = max(data)
    Traceback (most recent call last):
        raise ValueError("The array contains NaN values. The max function cannot be applied to arrays with NaN values.")
    ValueError: The array contains NaN values. The max function cannot be applied to arrays with NaN values.

    ------

    >>> import proadv as adv 
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> maximum = adv.statistics.descriptive.max(data)
    >>> maximum
    5

    ------



    >>> from proadv.statistics.descriptive import max 
    >>> import numpy as np
    >>> data = np.arange(2,10)
    >>> maximum = max(data)
    >>> maximum
    9
    """

    for i in data:
        if isinstance(i,
                      str):  # isinstance returns True if the specified object is of the specified type, otherwise False.
            raise TypeError("String cannot be placed as an element of an array")

    if np.isnan(data).any():
        raise ValueError("The array contains NaN values. The max function cannot be applied to arrays with NaN values.")

    if np.size(data) == 0:
        # The array cannot be empty
        raise ValueError("cannot calculate maximum with empty array")
    maximum = np.max(data)  # Calculate the maximum

    return maximum


def mean(array):
    """
    Calculate the mean of a dataset, handling non-numeric and numeric data.

    This function calculates the mean of a dataset
        ,which can be a single number, a list of numbers, or a list containing
        both numbers and strings. Non-numeric strings are converted to floats if possible, and ignored if not.

    Parameters
    ------
    array (numeric, array_like): The input data. Can be a numeric or array_like value.

    Returns
    ------
    mean_array (float | str): The mean of the input data if it is valid, otherwise a message indicating invalid input.

    Notes
    ------
    The function first checks if the input is a single number and returns it if so.
        If the input is a list,the function
        checks if all elements are numeric and calculates the mean.
        If there are strings in the list, it attempts to convert
        them to floats and calculates the mean of the numeric values.
        If the input is invalid or empty, it returns an error message.

    Examples
    ------
    >>> import proadv as adv 
    >>> import numpy as np
    >>> array = np.array([1, 2, 3])
    >>> mean_array = adv.statistics.descriptive.mean(array)
    >>> mean_array
    2.0

    ------

    >>> from proadv.statistics.descriptive import mean 
    >>> import numpy as np
    >>> array = np.array([1, 2, 3, 4, 5])
    >>> mean_array = mean(array)
    >>> mean_array
    3.0

    ------

    >>> from proadv.statistics.descriptive import mean 
    >>> import numpy as np
    >>> array = np.array([1, '2', 3.5, 'not a number'])
    >>> mean_array = mean(array)
    >>> mean_array
    2.1666666666666665

    ------

    >>> import proadv as adv 
    >>> import numpy as np
    >>> array = np.array([])
    >>> mean_array = adv.statistics.descriptive.mean(array)
    >>> mean_array
    'Invalid input.'
    """

    if isinstance(array, (int, float)):
        return array  # If it's a single number, return it as the mean

    elif np.size(np.array(array)) != 0 and type(array) != str and all(isinstance(item, (int, float)) for item in array):
        # The 'all()' function checks if all elements in 'data' satisfy a condition
        # The 'isinstance(item, (int, float))' inside 'all()' checks each item to ensure it is a number (int or float)
        # If any item is not a number, 'all()' returns False, and the function returns a message about non-num values
        array = list(array)
        mean_array = np.sum(array) / np.size(array)
        return mean_array  # Returns the average if all the items are an int or float

    # if at least one of the items in our list is not an int or float:
    if not isinstance(array, (int, float)) and np.size(np.array(array)) != 0:
        for item in array:
            if isinstance(item, str):
                # Trys to convert the string to a float
                try:
                    array = list(array)
                    index = array.index(item)
                    array[index] = float(item)

                except ValueError:
                    # If conversion fails, it's not a number, so we take out the invalid data and calculate the rest
                    array = [item for item in array if isinstance(item, (int, float))]
                    continue
        mean_array = np.sum(array) / np.size(array)
        return mean_array

    else:
        return "Invalid input."  # if there are any invalid inputs like a str this statement becomes true


def median(data):
    """
    Compute the median along the specified axis.

    Parameters
    ------
    data (array_like): The 1D array of data for which to calculate the Median.

    Returns
    ------
    med: the median of the data

    Raises
    ------
    TypeError: If the  element of array is a string.
    ValueError: If the array is empty. 

    Examples
    ------
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([14, 8, 11, 10, 5, 7])
    >>> med = adv.statistics.descriptive.median(data)
    >>> med
    9.0
    
    ------

    >>> from proadv.statistics.descriptive import median
    >>> import numpy as np
    >>> data = np.random.rand(15)
    >>> med = median(data) 

    ------

    >>> import proadv as adv
    >>> import numpy as np 
    >>> data = np.arange(14,45)
    >>> med = adv.statistics.descriptive.median(data)
    >>> med
    29.0
    """

    if data.ndim != 1:  # Optional check for 1D array
        raise ValueError("Data array must be a 1D array.")
    for i in data:
        if isinstance(i, str):
            # isinstance returns True if the specified object is of the specified type, otherwise False.
            raise TypeError("String cannot be placed as an element of an array")
    if np.size(data) == 0:
        # The array cannot be empty
        raise ValueError("cannot calculate median with empty array")
    med = np.median(data)  # Calculate the median
    return med


def mode(data):
    """
    Compute an array of the modal (most common) value in the passed array.

    Parameters
    ------
    data (array_like): The 1D array of data for which to calculate the Mode.

    Returns:
    ------
    mode_value (int): The mode of the data
    frequency (int): The number of repetitions of the mode

    Exampls
    ------
    
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([3, 0, 3, 7, 2])
    >>> values, counts = adv.statistics.descriptive.mode(data)
    >>> values
    3
    >>> counts
    2

    ------

    >>> from proadv.statistics.descriptive import mode
    >>> import numpy as np
    >>> data = np.array([4, 6, 12, 4, 15, 4, 6, 16])
    >>> values, counts = mode(data)
    >>> values
    4
    >>> counts
    3
    """
    values, counts = np.unique(data, return_counts=True)  # Unique values ​​and replicate counts are calculated
    max_count = np.argmax(counts)  # Calculate the indices of the maximum values ​​in the count array
    mode_value = values[max_count]  # values mc => max count
    frequency = counts[max_count]  # counts mc => max count
    return mode_value, frequency
