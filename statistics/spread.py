import numpy as np


def variance(data):
    """
    Compute the variance along the specified axis.

    The variance of the array elements, a measure of the spread of a distribution. 
    The variance is computed for the flattened array by default, otherwise over the specified axis.

    Parameters
    ------
    data (np.ndarray): The 1D array of data for which to calculate the variance. 
       

    Returns
    ------
    var : The Variance of the data
    
    Raises
    ------
    TypeError:
        "String cannot be placed as an element of an array."

    ValueError:
        "Data array must be a 1D array." 


    Examples
    ------
    >>> import proadv as dav # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.array([14, 8, 11, 10])
    >>> var = adv.statistics.spread.variance(data)
    >>> print(var)
    4.6875

    ------

    >>> from adv.statistics.spread import variance # Option 2: Direct import
    >>> import numpy as np
    >>> data = np.random.randn(20)
    >>> var = variance(data)

    ------

    >>> import proadv as dav # Option 1: Full import path
    >>> import numpy as np
    >>> data = np.arange(15,30)
    >>> var = adv.statistics.spread.variance(data)
    >>> print(var)
    18.666666666666668
    """

    if data.ndim != 1:  # Optional check for 1D array
        raise ValueError("Data array must be a 1D array.")    
    for i in data:
        if isinstance(i, str) :
        # isinstance returns True if the specified object is of the specified type, otherwise False.
            raise TypeError ("String cannot be placed as an element of an array")
    if np.size(data) == 0:
        # The array cannot be empty
        raise ValueError ("cannot calculate variance with empty array")
    var = np.var(data)
    return var



def std(data):
    # Implement the logic to calculate standard deviation of the data array
    pass
