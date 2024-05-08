# Variance Function

Return the `variance` along the specified axis. 
In this function:
1. If the element of array is a string, it raises a TypeError.
2. If the array is empty, it raises a ValueError.

- Examples:

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([14, 8, 11, 10])
    >>> var = adv.statistics.spread.variance(data)
    >>> var
    4.6875

>>>
    >>> from proadv.statistics.spread import variance
    >>> import numpy as np
    >>> data = np.random.randn(20)
    >>> var = variance(data)

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.arange(15,30)
    >>> var = adv.statistics.spread.variance(data)
    >>> var
    18.666666666666668


# STD Function

Return the `standard deviation` along the specified axis. 
If out is None, return a new array containing the standard deviation, otherwise return a reference to the output array. 
In this function:
1. If the element of array is a string, it raises a TypeError.
2. If the array is empty, it raises a ValueError.

- Examples:

>>>
    >>> from proadv.statistics.spread import std
    >>> import numpy as np
    >>> data = np.random.rand(25)
    >>> stdev = std(data)

>>>
    >>> import proadv as adv
    >>> import numpy as np
    >>> data = np.array([14, 8, 11, 10, 5, 7])
    >>> stdev = adv.statistics.spread.std(data)
    >>> stdev
    2.9107081994288304

>>>
    >>> import proadv as adv
    >>> import numpy as np  
    >>> data = np.arange(3,10)
    >>> stdev = adv.statistics.spread.std(data)
    >>> stdev
    2.0