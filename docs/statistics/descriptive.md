# Statistical Analysis: 
`ProADV` equips you with essential statistical tools to characterize your ADV data:

## Minimum

Return the `minimum` of an array input while checking for NaN values. If NaN values are present, it raises a ValueError. 
It also handles various exceptions that may occur during the operation.
The input data which should be an array or any array-like structure.
If the array contains NaN values, the function will not return a value and will raise a ValueError instead.
On the other hand, this function will raise a TypeError if the type of array is something other than integer.

- **Examples**:

```python
>>> import proadv as adv 
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5])
>>> minimum = adv.statistics.descriptive.min(data)
>>> minimum
1
```

```python
>>> from proadv.statistics.descriptive import min
>>> import numpy as np
>>> data = np.array([1, 2, np.nan, 4, 5])
>>> minimum = min(data)
Traceback (most recent call last):
    raise ValueError("The array contains NaN values. The min function cannot be applied to arrays with NaN values.")
ValueError: The array contains NaN values. The min function cannot be applied to arrays with NaN values.
```

```python
>>> from proadv.statistics.descriptive import min 
>>> import numpy as np
>>> data = np.random.rand(20) 
>>> minimum = min(data)
```


## Maximum

This function calculates the `maximum` value of an array-like input while checking for NaN values. If NaN values are 
present, it raises a ValueError. It also handles various exceptions that may occur during the operation.
The input data which should be an array or any array-like structure. 
If the array contains NaN values, the function will not return a value and will raise a ValueError instead.
On the other hand, this function will raise a TypeError if the type of array is something other than integer.

- **Examples**:

```python
>>> from proadv.statistics.descriptive import max
>>> import numpy as np
>>> data = np.array([1, 2, np.nan, 4, 5])
>>> maximum = max(data)
Traceback (most recent call last):
    raise ValueError("The array contains NaN values. The max function cannot be applied to arrays with NaN values.")
ValueError: The array contains NaN values. The max function cannot be applied to arrays with NaN values.
```

```python
>>> import proadv as adv 
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5])
>>> maximum = adv.statistics.descriptive.max(data)
>>> maximum
5
```

```python
>>> from proadv.statistics.descriptive import max 
>>> import numpy as np
>>> data = np.arange(2,10)
>>> maximum = max(data)
>>> maximum
9
```

```python
>>> import proadv as adv 
>>> import numpy as np
>>> data = np.array(["proadv"])
>>> maximum = adv.statistics.descriptive.max(data)
>>> maximum
Traceback (most recent call last):
    raise TypeError("String cannot be placed as an element of an array")
TypeError: String cannot be placed as an element of an array
```

```python
>>> from proadv.statistics.descriptive import max 
>>> import numpy as np
>>> data = np.array([])
>>> maximum = max(data)
>>> maximum
Traceback (most recent call last):
    raise ValueError("cannot calculate maximum with empty array")
ValueError: cannot calculate maximum with empty array
```


## Mean

Return the `mean` of a dataset, handling non-numeric and numeric data. Dataset can be a single number, a list of numbers, 
or a list containing both numbers and strings. Non-numeric strings are converted to floats if possible, and ignored if 
not. 

- **Notes**:

The function first checks if the input is a single number and returns it if so. If the input is a list,the function
checks if all elements are numeric and calculates the mean. If there are strings in the list, it attempts to convert
them to floats and calculates the mean of the numeric values. If the input is invalid or empty, it returns an error message.

- **Examples**:

```python
>>> import proadv as adv 
>>> import numpy as np
>>> array = np.array([1, 2, 3])
>>> mean_array = adv.statistics.descriptive.mean(array)
>>> mean_array
2.0
```

```python
>>> from proadv.statistics.descriptive import mean 
>>> import numpy as np
>>> array = np.array([1, 2, 3, 4, 5])
>>> mean_array = mean(array)
>>> mean_array
3.0
```

```python
>>> from proadv.statistics.descriptive import mean 
>>> import numpy as np
>>> array = np.array([1, '2', 3.5, 'not a number'])
>>> mean_array = mean(array)
>>> mean_array
2.1666666666666665
```

```python
>>> import proadv as adv 
>>> import numpy as np
>>> array = np.array([])
>>> mean_array = adv.statistics.descriptive.mean(array)
>>> mean_array
Invalid input
```


## Median

Returnn the `median` along the specified axis. If the element of array is a string, it raises a TypeError. 
Also, If the array is empty, it raises a ValueError.

- **Examples**:

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([14, 8, 11, 10, 5, 7])
>>> med = adv.statistics.descriptive.median(data)
>>> med
9.0
```

```python
>>> from proadv.statistics.descriptive import median
>>> import numpy as np
>>> data = np.random.rand(15)
>>> med = median(data) 
```

```python
>>> import proadv as adv
>>> import numpy as np 
>>> data = np.arange(14,45)
>>> med = adv.statistics.descriptive.median(data)
>>> med
29.0
```

```python
>>> import proadv as adv
>>> import numpy as np 
>>> data = np.array([])
>>> med = adv.statistics.descriptive.median(data)
>>> med
Traceback (most recent call last):
    raise ValueError("cannot calculate median with empty array")
ValueError: cannot calculate median with empty array
```

```python
>>> import proadv as adv
>>> import numpy as np 
>>> data = np.array([[1,2],[6,8]])
>>> med = adv.statistics.descriptive.median(data)
>>> med
Traceback (most recent call last):
    raise ValueError("Data array must be a 1D array.")
ValueError: Data array must be a 1D array.
```

```python
>>> from proadv.statistics.descriptive import median
>>> import numpy as np
>>> data = np.array(["proadv"])
>>> med = median(data) 
>>> med
Traceback (most recent call last):
    raise TypeError("String cannot be placed as an element of an array")
TypeError: String cannot be placed as an element of an array
```


## Mod

This function computes an array of the modal (most common) value in the passed array. 
It returns:
1. **mode_value (int)**: The mode of the data
2. **frequency (int)**: The number of repetitions of the mode

- **Examples**:

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([3, 0, 3, 7, 2])
>>> values, counts = adv.statistics.descriptive.mode(data)
>>> values
3
>>> counts
2
```

```python
>>> from proadv.statistics.descriptive import mode
>>> import numpy as np
>>> data = np.array([4, 6, 12, 4, 15, 4, 6, 16])
>>> values, counts = mode(data)
>>> values
4
>>> counts
3
```