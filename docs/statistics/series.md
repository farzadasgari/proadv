# Advanced Analysis: 
In addition to cleaning and basic statistics, `ProADV` offers advanced functionalities for deeper insights:

## Moving Average Function

`Moving Averages`, a statistical method in data analysis, smooths fluctuations in time-series data to reveal underlying 
trends. Calculating the average within a specified window and shifting it through the dataset, provides a clearer trend 
representation. Widely applied in finance, economics, and signal processing, Moving Averages come in types like Simple 
Moving Average (SMA) and Exponential Moving Average (EMA), each with unique weighting methods for data points.
A `moving average` is a technical indicator that investors and traders use to determine the trend direction of 
securities. It is calculated by adding up all the data points during a specific period and dividing the sum by the number 
of time periods.
This function return the `moving average` of a 1D array and implements a cumulative moving average calculation. It's more
efficient than calculating a simple moving average for each element. 

There are two parameters in this function:
1. **data (array_like)**: The 1D array of data for which to calculate the moving average.
2. **window_size (int, optional)**: The size of the window for the moving average. Defaults to 20. Must be less than or equal to the size of the data array.
If the window size is larger than the size of the data array, it raises a ValueError. 

- **plot**:


- **Examples**:

```python
>>> from proadv.statistics.series import moving_average
>>> import numpy as np
>>> data = np.random.rand(300)
>>> ma = moving_average(data)
```

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> ma = adv.statistics.series.moving_average(data, window_size=3)
>>> ma
array([1. , 1.5, 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ])
```

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([])
>>> ma = adv.statistics.series.moving_average(data, window_size=3)
>>> print(ma)
[]
```

```python
>>> from proadv.statistics.series import moving_average
>>> import numpy as np
>>> data = np.array([[4,9],[8,6]])
>>> ma = moving_average(data)
>>> print(ma)
Traceback (most recent call last):
    raise ValueError("Data array must be a 1D array.")
ValueError: Data array must be a 1D array.
```

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5, 6, 7])
>>> ma = adv.statistics.series.moving_average(data, window_size=-3)
>>> print(ma)
Traceback (most recent call last):
    raise ValueError("Window size must be greater than zero.")
ValueError: Window size must be greater than zero.
```

```python
>>> from proadv.statistics.series import moving_average
>>> import numpy as np
>>> data = np.array([1, 2, 3, 5, 7])
>>> ma = moving_average(data , window_size=7)
>>> print(ma)
Traceback (most recent call last):
    raise ValueError("Data array size must be greater than window size.")
ValueError: Data array size must be greater than window size.
```


## Exponential Moving Average Function

The exponential moving average (EMA) is a type of moving average that places more weight on recent observations while
still considering older data. It is particularly useful for smoothing noisy data and identifying trends.
`Exponential Moving Average` is calculated by taking the weighted mean of the observations at a time. The weight of the 
observation exponentially decreases with time. It is used for analyzing recent changes.
This function calculates the `exponential moving average` of a given data series. 

In this function There is a loop that calculates the exponential moving average for elements from index 1 onwards.
It utilizes the previously calculated exponential moving average (ema[i-1]) and the new data point
(data[i]) to efficiently update the exponential moving average using the formula:
`ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]`

Update the exponential moving average based on the previous average, new data point, and the smoothing factor alpha.

There are two parameters in this function:

1. **data (array_like)**: The 1D array of data for which to calculate the exponential moving average.
2. **alpha (float, optional)**: Smoothing factor between 0 and 1. Higher alpha discounts older observations faster. Default is 0.2.
If alpha is not between 0 and 1 (inclusive), It raises a ValueError. 

- **plot**:


- **Example**:

```python
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> ema = exponential_moving_average(data, alpha=0.5)
>>> ema
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```


## Weighted Moving Average Function
 
The weighted moving average (WMA) is a type of moving average that assigns a greater weighting to the most recent data points, and less weighting to data points in the distant past.
To implement a weighted moving average in Python, you can create a function that takes a list of values and their corresponding weights as inputs.

The formula of weighted moving average is as follows:

![formula](https://community.esri.com/t5/image/serverpage/image-id/26805iE2664703F829F856?v=v2)

This function calculates the `weighted moving average` of a 1D array.

There are two parameters in this function:
1. **data (array_like)**: The 1D array of data for which to calculate the weighted moving average.
2. **period (int, optional)**: The period for the weighted moving average. Defaults to 20. Must be less than or equal to the size of the data array.
If the period is larger than the size of the data array, it raises a ValueError. 

- **plot**:


- **Examples**:

```python
>>> import proadv as adv  
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> wma = adv.statistics.series.weighted_moving_average(data, period=3)
>>> wma
array([2.33333333, 3.33333333, 4.33333333, 5.33333333, 6.33333333,
       7.33333333, 8.33333333])
```

```python
>>> from proadv.statistics.series import weighted_moving_average  
>>> import numpy as np
>>> data = np.random.rand(300)
>>> wma = weighted_moving_average(data)
```

```python
>>> from proadv.statistics.series import weighted_moving_average  
>>> import numpy as np
>>> data = np.array([1, 2, 4, 6, 8])
>>> wma = weighted_moving_average(data, period=8)
>>> print(wma)
Traceback (most recent call last):
    raise ValueError("Data array size must be greater than period.")
ValueError: Data array size must be greater than period.
```

```python
>>> import proadv as adv  
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> wma = adv.statistics.series.weighted_moving_average(data, period=-3)
>>> print(wma)
Traceback (most recent call last):
    raise ValueError("Period must be greater than zero.")
ValueError: Period must be greater than zero.
```


## _mobility Function

Return the covert `mobility` index.
In this function, The obfuscated mobility value, representing the hidden patterns in the data.
The parameter for this functioni is the array containing clandestine data. 
Input array must also have at least two elements, otherwise, it will raise a ValueError


## _diagonal_average Function

Calculate the `diagonal average` of a matrix with complex patterns.

There are three parameters in this function:
1. **matrix (array_like)**: Input matrix with hidden patterns.
2. **length (int)**: Length parameter affecting the calculation.
3. **size (int)**: Size parameter affecting the calculation.


## SSA Function

Subspace based techniques are used to enhance the noise corrupted time series signals. 
`Singular spectrum analysis (SSA)` is a subspace based method that decomposes the time series data into the
trend, oscillating and noise components. It is widely used for analysing climatic and biomedical signals. 
The main difference between the traditional SSA method and our method lies in the way of identifying the 
desired signal (the velocity signal) subspace; in other words the grouping step in SSA. 
In traditional SSA, the desired signal subspace is estimated by based on the magnitude of the eigenvalues. However, in
our technique, the desired signal subspace is estimated based on the local mobility of the eigenvectors. 
SSA includes four basic steps: embedding, decomposition, grouping and reconstruction.

Perform `Singular Spectrum Analysis (SSA)` on a given signal. It returns filtered signal after SSA.

There are three parameters in this function:
1. **x (array_like)**: Input signal.
2. **fs (float/int)**: Sampling frequency of the signal.
3. **f (float/int)**: maximum frequency of the signal of interest.

- **Reference**:

Sharma, Anurag, Ajay Kumar Maddirala, and Bimlesh Kumar.
"Modified singular spectrum analysis for despiking acoustic Doppler velocimeter (ADV) data."
Measurement 117 (2018): 339-346.


## Kalman Filter

Oceanic turbulence measurements made by an acoustic Doppler velocimeter (ADV) suffer from noise that potentially 
affects the estimates of turbulence statistics. This function examines the abilities of `Kalman filtering` model to 
eliminate noise in ADV velocity datasets of laboratory experiments and offshore observations. 

The `Kalman filter` is an algorithm that tracks an optimal estimate of the state of a stochastic dynamical system, 
given a sequence of noisy observations or measurements of the state over time.
This function calculates `kalman filter` for a 1D array. 

- **plot**:


- **Parameters**:

There are five parameters in the Kalman Filter function:

- **data (array_like)**: The 1D array of data for which to calculate the kalman filter.
- **initial_state (array_like)**: An initial estimate for the state variable.
- **initial_covariance (array_like)**: An initial estimate for the covariance.
- **process_noise (array_like)**: Process noise that occurs in the process of changing a state variable.
- **measurement_noise (array_like)**: Measurement noise present in the input data.

- **Reference**:

Huang, Chuanjiang, Fangli Qiao, and Hongyu Ma. "Noise reduction of acoustic Doppler velocimeter data based on Kalman
filtering and autoregressive moving average models." Acta Oceanologica Sinica 39 (2020): 106-113.

- **plot**:

- **Examples**:

```python
>>> from proadv.statistics.series import kalman_filter  
>>> import numpy as np
>>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> initial_state = np.array([[0]])
>>> initial_covariance = np.array([[1]])
>>> process_noise = np.array([[0.001]])
>>> measurement_noise = np.array([[10]])
>>> filtered_data = kalman_filter(data, initial_state, initial_covariance, process_noise, measurement_noise)
>>> print(filtered_data)
[array([[0.09099173]]), array([[0.25036867]]), array([[0.46247241]]), array([[0.71611623]]), array([[1.00309733]]),
 array([[1.31726603]]), array([[1.65392287]]), array([[2.00941642]]), array([[2.38086783]]), array([[2.76597798]])]
```

```python
>>> import proadv as adv
>>> import numpy as np
>>> data = np.array([9, 5, 7, 3, 6, 4, 1.5])
>>> initial_state = np.array([[10]])
>>> initial_covariance = np.array([[10]])
>>> process_noise = np.array([[0.001]])
>>> measurement_noise = np.array([[15]])
>>> filtered_data = adv.statistics.series.kalman_filter(data, initial_state, initial_covariance, process_noise,        measurement_noise)
>>> print(filtered_data)
[array([[9.599976]]), array([[8.28548437]]), array([[7.99973337]]), array([[7.09023925]]),
 array([[6.92238759]]), array([[6.53234292]]), array([[5.93951465]])]
```
