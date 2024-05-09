# Replacement Methods: 
`ProADV` offers several options to replace detected spikes with more reliable values:


## LVD (Last Valid Data) Function

Replace spike values in velocities array with the `last valid data` before each. 
It returns modified data with spikes replaced by last valid values. An array containing the modified data.

This function has two parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.


## MV (Mean Value) Function

Replace spike values in velocities array with the `mean value` of velocity component. 
It returns modified data with spikes replaced by mean value of velocity component. An array containing the modified data.

This function has two parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.


## LI (Linear Interpolation) Function

This function identifies spikes in the velocity data and replaces them with `linearly interpolated values` based on the nearest valid data points before and after each spike. 
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