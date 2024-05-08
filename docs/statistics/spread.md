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