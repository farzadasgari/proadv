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
    """
    Compute the standard deviation along the specified axis.

    Standard deviation, a measure of the spread of a distribution, 
    of the array elements. The standard deviation is computed for the flattened array 
    by default, otherwise over the specified axis.

    Parameters
    ------
    data (np.ndarray): The 1D array of data for which to calculate the Standard deviation. 

    Returns
    ------
    stdev : The standard deviation of data. 
    If out is None, return a new array containing the standard deviation, 
    otherwise return a reference to the output array. 

    Raises
    ------
    TypeError
        "String cannot be placed as an element of an array."

    ValueError:
        "Data array must be a 1D array." 


    Examples
    ------
    >>> import proadv as adv  # Option 1: Full import path
    >>> import numpy as np  
    >>> data = np.array([14, 8, 11, 10, 5, 7])
    >>> stdev = adv.statistics.spread.std(data)
    >>> print(stdev)
    2.9107081994288304

    ------
    
    >>> from proadv.statistics.spread import std  # Option 2: Direct import
    >>> import numpy as np
    >>> data = np.random.rand(25)
    >>> stdev = std(data)

    ------

    >>> import proadv as adv  # Option 1: Full import path
    >>> import numpy as np  
    >>> data = np.arange(3,10)
    >>> stdev = adv.statistics.spread.std(data)
    >>> print(stdev)
    2.0
    """
    
    if data.ndim != 1:  # Optional check for 1D array
        raise ValueError("Data array must be a 1D array.")
    
    for element in data:
        if isinstance(element , str):
        # isinstance returns True if the specified object is of the specified type, otherwise False. 
            raise TypeError ("String cannot be placed as an element of an array")
    if np.size(data) == 0:
        # The array cannot be empty
        raise ValueError ("cannot calculate standard deviation with empty array")    
    stdev= np.std(data)
    return stdev
