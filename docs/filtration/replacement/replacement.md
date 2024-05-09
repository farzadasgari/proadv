# Replacement Methods: 
`ProADV` offers several options to replace detected spikes with more reliable values:


## LVD (Last Valid Data) Function

Replace spike values in velocities array with the `last valid data` before each. 
It returns modified data with spikes replaced by last valid values. An array containing the modified data.

This function has three parameters:
1. **velocities (array_like)**: Array of velocity data. An array-like object containing velocity values.
2. **spike_indices (array_like)**: Indices of spikes. An array-like object containing the indices of detected spikes.

