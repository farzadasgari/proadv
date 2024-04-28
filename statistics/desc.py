import numpy as np


def min():
    # Implement the logic to find the minimum value in the data array
    pass


def max():
    # Implement the logic to find the maximum value in the data array
    pass


def mean():
    # Implement the logic to calculate the mean of the data array
    pass


def median(data):
    '''
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    -------
    data (np.ndarray): The 1D array of data for which to calculate the Median. 

    Returns
    -------
    Calculate the median of the data

    Raises
    -------
    TypeError
        "String cannot be placed as an element of an array"

    ValueError
        "cannot calculate median with empty array"

    Examples
    -------
    >>> data =  np.array([14, 8, 11, 10, 5, 7])
    >>> np.median(data)
    Out : 9.0

    '''
    try:
        if data.ndim != 1:  # Optional check for 1D array
            raise ValueError("Data array must be a 1D array.")

        if isinstance(data, str) :
            # isinstance returns True if the specified object is of the specified type, otherwise False.
            raise TypeError ("String cannot be placed as an element of an array")
        if np.size(data) == 0:
            # The array cannot be empty
            raise ValueError ("cannot calculate median with empty array")
        _median_ = np.median(data) # Calculate the median
        return _median_
    except TypeError as TE :
        print(f"Type Error: {TE}")
    except ValueError as VE:
        print(f"Value Error:{VE}")
