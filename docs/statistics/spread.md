# Variance 

Compute the `Variance` along the specified axis of a 1D array. 
In this function:
1. If the element of array is a string, it raises a TypeError.
2. If the array is empty, it raises a ValueError.

- **Examples**

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([14, 8, 11, 10])
>>> var = adv.statistics.spread.variance(data)
>>> var
4.6875
```
<br>

```python
>>> from proadv.statistics.spread import variance
>>> import numpy as np
>>> data = np.random.randn(20)
>>> var = variance(data)
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.arange(15,30)
>>> var = adv.statistics.spread.variance(data)
>>> var
18.666666666666668
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([])
>>> var = adv.statistics.spread.variance(data)
>>> var
Traceback (most recent call last):
    raise ValueError("cannot calculate variance with empty array")
ValueError: cannot calculate variance with empty array
```
<br>

```python
>>> from proadv.statistics.spread import variance
>>> import numpy as np
>>> data = np.array([[1,2],[9,3]])
>>> var = variance(data)
>>> var
Traceback (most recent call last):
    raise ValueError("Data array must be a 1D array.")
ValueError: Data array must be a 1D array.
```
<br>

```python
>>> from proadv.statistics.spread import variance
>>> import numpy as np
>>> data = np.array(["proadv"])
>>> var = variance(data)
>>> var
Traceback (most recent call last):
    raise TypeError("String cannot be placed as an element of an array")
TypeError: String cannot be placed as an element of an array
```


# Standard Deviation

Compute the `Standard Deviation` along the specified axis of a 1D array. 
If out is None, return a new array containing the standard deviation, otherwise return a reference to the output array. 
In this function:
1. If the element of array is a string, it raises a TypeError.
2. If the array is empty, it raises a ValueError.

- **Examples**

```python
>>> from proadv.statistics.spread import std
>>> import numpy as np
>>> data = np.random.rand(25)
>>> stdev = std(data)
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([14, 8, 11, 10, 5, 7])
>>> stdev = adv.statistics.spread.std(data)
>>> stdev
2.9107081994288304
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np  
>>> data = np.arange(3,10)
>>> stdev = adv.statistics.spread.std(data)
>>> stdev
2.0
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np  
>>> data = np.array([])
>>> stdev = adv.statistics.spread.std(data)
>>> stdev
Traceback (most recent call last):
    raise ValueError("cannot calculate standard deviation with empty array")
ValueError: cannot calculate standard deviation with empty array
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np  
>>> data = np.array(["proadv"])
>>> stdev = adv.statistics.spread.std(data)
>>> stdev
Traceback (most recent call last):
    raise TypeError("String cannot be placed as an element of an array")
TypeError: String cannot be placed as an element of an array
```
<br>

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([[14, 8, 11],[4, 6, 8]])
>>> stdev = adv.statistics.spread.std(data)
>>> stdev
Traceback (most recent call last):
    raise ValueError("Data array must be a 1D array.")
ValueError: Data array must be a 1D array.
```
