# Replacement Methods
`ProADV` offers several options to replace detected spikes with more reliable values:


## LVD (Last Valid Data)

Replace spike values in velocities array with the `last valid data` before each. 
It returns modified data with spikes replaced by last valid values. An array containing the modified data.

This function has two parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.


## MV (Mean Value)

Replace spike values in velocities array with the `mean value` of velocity component. 
It returns modified data with spikes replaced by mean value of velocity component. An array containing the modified data.

This function has two parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.


## LI (Linear Interpolation)

This function replaces detected spikes with `linearly interpolated values` based on the nearest valid data points before and after each spike. 
If a spike occurs at the end of the data, it is replaced with the mean of the entire dataset. 
The function is robust to handle individual spikes as well as sequences of consecutive spikes.

It returns An array containing the modified velocity data with spikes replaced by interpolated values. 
The shape and type of the array are the same as the input 'velocities' array.

This function has three parameters:
1. **velocities** : (array_like)
    An array-like object containing velocity values. It should be a one-dimensional
    array of numerical data representing velocities.
2. **spike_indices** : (array_like)
    An array-like object containing the indices of detected spike events. It should
    be a one-dimensional array of integers where each integer represents the index
    in 'velocities' that corresponds to a spike.
3. **decimals** : int, optional
    The number of decimal places to round the interpolated values to. This allows
    the output data to be presented with a consistent level of precision. The default
    value is 4, but this can be adjusted as needed.

**Notes**
The function uses linear interpolation to estimate the values of spikes based on the
surrounding non-spike data. For spikes at the beginning or end of the data where a
neighboring non-spike value is not available, the function uses the mean of the entire
dataset as a replacement value. This approach ensures that the data remains as accurate
and representative of the original dataset as possible.


## 12PP (12 Points Cubic Polynomial)

Interpolates missing data points in a velocity array using `cubic polynomial interpolation`. 

It returns array of velocities with missing data interpolated. The interpolated values are calculated based on cubic polynomial interpolation using neighboring data points.

If spike_indices contains invalid indices or if velocities is not a one-dimensional array-like object, it raises a ValueError.

This function has three parameters:
1. **velocities (array_like)**: Array of velocity data. It should be a one-dimensional array-like object.
    This function assumes the input velocities array has at least 25 data points.
2. **spike_indices (array_like)**: Indices where data is missing (spikes). It should be a one-dimensional array-like
    object containing integers representing the indices of missing data points (spikes).
    If spike_indices contains invalid indices (e.g., negative values or indices exceeding the array size),
    a ValueError will be raised.
3. **decimals (int, optional)**: Number of decimal places to round the result to. Default is 4.

**Notes**:
- This function uses cubic polynomial interpolation to estimate missing data points based on neighboring values.
- If a missing data point (spike) occurs near the boundaries of the velocities array, linear interpolation is used instead of cubic polynomial interpolation.
- The input velocities array is modified in-place to replace missing data points with interpolated values.


## Simple Moving Average

Replace spike values in velocities array with the `Moving Average` of main data. 
It returns modified data with spikes replaced by moving average of main data. An array containing the modified data.

This function has three parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.
3. **window_size (int, optional)**: The size of the window for the moving average. This is a desired value. Defaults to 20. Must be less than or equal to the         size of the data array. If the window size is larger than the size of the data array, it raises a ValueError.


## Exponential Moving Average

This function replaces the spike values in velocities array with the `Exponential Moving Average` values of the main data.
It returns modified data with spikes replaced by exponential moving average of main data. An array containing the modified data.

This function has three parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.
3. **alpha (float, optional)**: Smoothing factor between 0 and 1. Higher alpha discounts older observations faster. Default is 0.2. If alpha is not between 0        and 1 (inclusive), It raises a ValueError.


## Weighted Moving Average

Replace spike values in velocities array with the `Weighted Moving Average` of main data. 
It returns modified data with spikes replaced by weighted moving average of main data. An array containing the modified data.

This function has three parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.
3. **period (int, optional)**: The period for the weighted moving average. Defaults to 20. Must be less than or equal to the size of the data array. If the         period is larger than the size of the data array, it raises a ValueError.


## Singular Spectrum Analysis

This function replaces detected spikes with `Singular Spectrum Analysis` values. 
It returns modified data with spikes replaced by singular spectrum analysis. An array containing the modified data.

This function has four parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.
3. **fs (float/int)**: Sampling frequency of the signal.
4. **f (float/int)**: maximum frequency of the signal of interest.


## Kalman

Replace spike values in velocities array with the `Kalman` values. 
It returns modified data with spikes replaced by kalman values. An array containing the modified data.

This function has six parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.
3. **initial_state (array_like)**: An initial estimate for the state variable.
4. **initial_covariance (array_like)**: An initial estimate for the covariance.
5. **process_noise (array_like)**: Process noise that occurs in the process of changing a state variable.
6. **measurement_noise (array_like)**: Measurement noise present in the input data.









