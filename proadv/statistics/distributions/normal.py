import numpy as np
def cdf(array, decimals=4):
    from math import erf, erfc
    array = np.copy(array)
    if array.size == 0:
        raise ValueError("cannot calculate PDF with empty array.")
    if np.isnan(array).any():
        raise TypeError('array cannot contain nan values.')
    np_sqrt = 1.0 / np.sqrt(2)
    array_ns = array * np_sqrt
    absolute_value = np.fabs(array_ns)
    j = 0
    for i in absolute_value:
        if i < np_sqrt:
            array[j] = 0.5 + 0.5 * erf(array_ns[j])
        else:
            y = 0.5 * erfc(i)
            if array_ns[j] > 0:
                array[j] = 1.0 - y
            else:
                array[j] = y
        j += 1
    return np.around(array, decimals=decimals)
