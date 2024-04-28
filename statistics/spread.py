import numpy as np


def variance(data):
    '''
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a distribution. 
    The variance is computed for the flattened array by default, otherwise over the specified axis.

    Parameters
    -------
    data (np.ndarray): The 1D array of data for which to calculate the variance. 
       

    Returns
    -------
    _var_
    
    Raises
    -------
    TypeError:
        "String cannot be placed as an element of an array."

    ValueError:
        "Data array must be a 1D array." 


    Examples
    -------
    >>> data = [14, 8, 11, 10]
    >>> variance(data)
    >>> out : 4.6875

    '''
    try :
        if data.ndim != 1:  # Optional check for 1D array
            raise ValueError("Data array must be a 1D array.")
        _var_ = np.var(data)
        
        for element in data:
            if isinstance(element , str):
            # isinstance returns True if the specified object is of the specified type, otherwise False. 
                raise TypeError ("String cannot be placed as an element of an array")
        return _var_
            
    except TypeError as TE:
        print(f"Type Error: {TE}")
    except ValueError as VE:
        print(f"Value Error:{VE}")


def std(data):
    '''
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution, 
    of the array elements. The standard deviation is computed for the flattened array 
    by default, otherwise over the specified axis.

    Parameters
    -------
    data (np.ndarray): The 1D array of data for which to calculate the Standard deviation. 

    Returns
    -------
    _std_ : ndarray
        If out is None, return a new array containing the standard deviation, 
        otherwise return a reference to the output array. 

    Raises
    -------
    TypeError
        "String cannot be placed as an element of an array."

    ValueError:
        "Data array must be a 1D array." 


    Examples
    -------
    >>> data = [14, 8, 11, 10, 5, 7]
    >>> std(data)
    >>> out : 2.9107081994288304
    '''
    try :
        if data.ndim != 1:  # Optional check for 1D array
            raise ValueError("Data array must be a 1D array.")
        _std_ = np.std(data)
        for element in data:
            if isinstance(element , str):
            # isinstance returns True if the specified object is of the specified type, otherwise False. 
                raise TypeError ("String cannot be placed as an element of an array")
        return _std_
    except TypeError as TE:
        print(f"Type Error: {TE}")
    except ValueError as VE:
        print(f"Value Error:{VE}")
