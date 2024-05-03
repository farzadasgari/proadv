import numpy as np


def min(x):
    """
        Calculate the minimum value in an array, handling NaN values and exceptions.

        This function calculates the minimum value of an array-like input while checking for NaN values.
        If NaN values are present, it raises a ValueError. It also handles various exceptions that may
        occur during the operation.

        Parameters
        ------
        x : array_like: The input data which should be an array or any array-like structure.


        Returns
        ------
        number: The minimum value in the array. If the array contains NaN values, the function will not return a value
                and will raise a ValueError instead.

        Raises
        ------
        TypeError: If an operation or function is applied to an object of inappropriate type.
        ValueError: If a function receives an argument of correct type but inappropriate value.
        AttributeError: If an attribute reference or assignment fails.
        IndexError: If a sequence subscript is out of range.
        MemoryError: If an operation runs out of memory.

        Examples
        ------
        >>> import proadv as adv  # Option 1: Full import path
        >>> import numpy as np
        >>> adv.statistics.desc.min([1, 2, 3, 4, 5])
        out: 1

        >>> from proadv.statistics.desc import min # Option 2: Direct import
        >>> import numpy as np
        >>> min(np.array([1, 2, np.nan, 4, 5]))
        ValueError: The array contains NaN values. The min function cannot be applied to arrays with NaN values.
        """

    try:
        if np.isnan(x).any():
            raise ValueError(
                "The array contains NaN values. The max function cannot be applied to arrays with NaN values.")
        return np.min(x)

    except TypeError as te:
        raise TypeError(f"Type Error: {te}")
    except ValueError as ve:
        raise ValueError(f"Value Error: {ve}")
    except AttributeError as ae:
        raise AttributeError(f"Attribute Error: {ae}")
    except IndexError as ie:
        raise IndexError(f"Index Error: {ie}")
    except MemoryError as me:
        raise MemoryError(f"Memory Error: {me}")


def max(x):
    """
     Calculate the maximum value in an array, handling NaN values and exceptions.

     This function calculates the maximum value of an array-like input while checking for NaN values.
     If NaN values are present, it raises a ValueError. It also handles various exceptions that may
     occur during the operation.

     Parameters
     ------
     x : array_like : The input data which should be an array or any array-like structure.

     Returns
     ------
     number :The maximum value in the array. If the array contains NaN values, the function will not return a value
             and will raise a ValueError instead.

     Raises
     ------
     TypeError: If an operation or function is applied to an object of inappropriate type.
     ValueError: If a function receives an argument of correct type but inappropriate value.
     AttributeError: If an attribute reference or assignment fails.
     IndexError: If a sequence subscript is out of range.
     MemoryError: If an operation runs out of memory.

     Examples
     ------
     >>> import proadv as adv  # Option 1: Full import path
     >>> import numpy as np
     >>> adv.statistics.desc.max([1, 2, 3, 4, 5])
     out: 5

     >>> from proadv.statistics.desc import max # Option 2: Direct import
     >>> import numpy as np
     >>> max(np.array([1, 2, np.nan, 4, 5]))
     ValueError: The array contains NaN values. The max function cannot be applied to arrays with NaN values.

     """

    try:
        if np.isnan(x).any():
            raise ValueError(
                "The array contains NaN values. The max function cannot be applied to arrays with NaN values.")
        return np.max(x)

    except TypeError as te:
        raise TypeError(f"Type Error: {te}")
    except ValueError as ve:
        raise ValueError(f"Value Error: {ve}")
    except AttributeError as ae:
        raise AttributeError(f"Attribute Error: {ae}")
    except IndexError as ie:
        raise IndexError(f"Index Error: {ie}")
    except MemoryError as me:
        raise MemoryError(f"Memory Error: {me}")


def mean(x):
    """
        Calculate the mean of a dataset, handling non-numeric and numeric data.

        This function calculates the mean of a dataset
        ,which can be a single number, a list of numbers, or a list containing
        both numbers and strings. Non-numeric strings are converted to floats if possible, and ignored if not.

        Parameters
        ------
        x : numeric, array_like : The input data. Can be a numeric or array_like value.

        Returns
        ------
        float or str: The mean of the input data if it is valid, otherwise a message indicating invalid input.

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
        >>> import proadv as adv  # Option 1: Full import path
        >>> import numpy as np
        >>> adv.statistics.desc.mean(5)
        out: 5

        >>> from proadv.statistics.desc import mean # Option 2: Direct import
        >>> import numpy as np
        >>> mean([1, 2, 3, 4, 5])
        out: 3.0

        >>> mean([1, '2', 3.5, 'not a number'])
        out: 2.1666666666666665

        >>> mean([])
        out: 'Invalid input.'
    """

    if isinstance(x, (int, float)):
        return x  # If it's a single number, return it as the mean

    elif np.size(np.array(x)) != 0 and type(x) != str and all(isinstance(item, (int, float)) for item in x):
        # The 'all()' function checks if all elements in 'data' satisfy a condition
        # The 'isinstance(item, (int, float))' inside 'all()' checks each item to ensure it is a number (int or float)
        # If any item is not a number, 'all()' returns False, and the function returns a message about non-num values
        x = list(x)
        return np.sum(x) / np.size(x)  # Returns the average if all the items are an int or float

    # if at least one of the items in our list is not an int or float:
    if not isinstance(x, (int, float)) and np.size(np.array(x)) != 0:
        for item in x:
            if isinstance(item, str):
                # Trys to convert the string to a float
                try:
                    x = list(x)
                    index = x.index(item)
                    x[index] = float(item)

                except ValueError:
                    # If conversion fails, it's not a number, so we take out the invalid data and calculate the rest
                    x = [item for item in x if isinstance(item, (int, float))]
                    continue

        return np.sum(x) / np.size(x)

    else:
        return "Invalid input."  # if there are any invalid inputs like a str this statement becomes true


def median(data):
    """
    Compute the median along the specified axis.

    Parameters
    ------
    data (np.ndarray): The 1D array of data for which to calculate the Median. 

    Returns
    ------
    med: the median of the data

    Raises
    ------
    TypeError: If the  element of array is a string.
    ValueError: If the array is empty. 

    Examples
    ------
    >>> import proadv as adv  # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.array([14, 8, 11, 10, 5, 7])
    >>> med = adv.statistics.desc.median(data) 
    >>> print(med)
    9.0
    
    ------

    >>> from proadv.statistics.desc import median # Option 2: Direct import
    >>> import numpy as np
    >>> data = np.random.rand(15)
    >>> med = median(data) 

    ------

    >>> import proadv as adv  # Option 1: Full import path
    >>> import numpy as np 
    >>> data = np.arange(14,45)
    >>> med = adv.statistics.desc.median(data)
    >>> print(med)
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
