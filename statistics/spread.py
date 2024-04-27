import numpy as np


def variance(data):
    '''
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a distribution. 
    The variance is computed for the flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    data (np.ndarray): The 1D array of data for which to calculate the variance. 
       

    Returns
    -------
    _var_
    
    Raises
    ------
    TypeError
        "String cannot be placed as an element of an array"

    Examples
    ------
    >>> data =  np.array([14, 8, 11, 10])
    >>> np.var(data)
    out : 4.6875

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
    # Implement the logic to calculate standard deviation of the data array
    pass
