import numpy as np
from proadv.statistics.descriptive import mean


def last_valid_data(velocities, spike_indices):
    """
    Replace spike values in velocities array with the last valid data before each.

    Parameters
    ------
        velocities (numpynd.array): Array of velocity data.
            An array-like object containing velocity values.
        spike_indices (numpy.ndarray): Indices of spikes.
            An array-like object containing the indices of detected spikes.

    Returns
    ------
        modified_data (numpy.ndarray): Modified data with spikes replaced by last valid values.
            An array containing the modified data.
    """
    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Replace values at spikes indices with the last valid value
    for idx in spike_indices:
        modified_data[idx] = modified_data[idx - 1]

    return modified_data


def mean_value(velocities, spike_indices):
    """
    Replace spike values in velocities array with the mean value of velocity component.

    Parameters
    ———
        velocities (numpynd.array): Array of velocity data.
            An array-like object containing velocity values.
        spike_indices (numpy.ndarray): Indices of spikes.
            An array-like object containing the indices of detected spikes.

    Returns
    ———
        modified_data (numpy.ndarray): Modified data with spikes replaced by mean value of velocity component.
            An array containing the modified data.
    """
    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Replace values at spikes indices with the mean value
    modified_data[spike_indices] = mean(velocities)

    return modified_data


def linear_interpolation(velocities, spike_indices, decimals=4):
    """
    Perform linear interpolation to replace spike values in velocity data.

    This function identifies spikes in the velocity data and replaces them with
    linearly interpolated values based on the nearest valid data points before
    and after each spike. If a spike occurs at the end of the data, it is replaced
    with the mean of the entire dataset. The function is robust to handle individual
    spikes as well as sequences of consecutive spikes.

    Parameters
    ------
    velocities : (numpy.ndarray)
        An array-like object containing velocity values. It should be a one-dimensional
        array of numerical data representing velocities.
    spike_indices : (numpy.ndarray)
        An array-like object containing the indices of detected spike events. It should
        be a one-dimensional array of integers where each integer represents the index
        in 'velocities' that corresponds to a spike.
    decimals : int, optional
        The number of decimal places to round the interpolated values to. This allows
        the output data to be presented with a consistent level of precision. The default
        value is 4, but this can be adjusted as needed.

    Returns
    ------
    modified_data (numpy.ndarray):
        An array containing the modified velocity data with spikes replaced by
        interpolated values. The shape and type of the array are the same as the
        input 'velocities' array.

    Notes
    ------
    The function uses linear interpolation to estimate the values of spikes based on the
    surrounding non-spike data. For spikes at the beginning or end of the data where a
    neighboring non-spike value is not available, the function uses the mean of the entire
    dataset as a replacement value. This approach ensures that the data remains as accurate
    and representative of the original dataset as possible.

    Examples
    ------
    >>> velocities = np.array([5, 6, 7, 50, 7, 6, 5])
    >>> spike_indices = np.array([3])
    >>> linear_interpolation(velocities, spike_indices)
    array([5., 6., 7., 7., 7., 6., 5.])
    """

    # Create a copy of the velocity data to avoid modifying the original array
    modified_data = np.copy(velocities)

    # Initialize spike values with NaN to facilitate interpolation
    modified_data[spike_indices] = np.nan

    # Calculate differences between consecutive spike indices
    spike_diff = np.diff(spike_indices)
    # Identify the start index of each spike sequence
    spike_starts = spike_indices[np.insert(spike_diff > 1, 0, True)]
    # Identify the end index of each spike sequence
    spike_ends = spike_indices[np.append(spike_diff > 1, True)]

    # Iterate over each spike sequence for interpolation
    for start, end in zip(spike_starts, spike_ends):
        # Find the valid data point immediately before the spike sequence
        valid_start = modified_data[start - 1] if start > 0 else np.nan
        # Find the valid data point immediately after the spike sequence
        valid_end = modified_data[end + 1] if end < len(velocities) - 1 else np.nan

        # If valid start and end points are found, perform linear interpolation
        if not np.isnan(valid_start) and not np.isnan(valid_end):
            # Calculate the interpolation step size
            step = (valid_end - valid_start) / (end - start + 2)
            # Apply linear interpolation across the spike sequence
            for i, idx in enumerate(range(start, end + 1)):
                modified_data[idx] = valid_start + step * (i + 1)
        elif np.isnan(valid_end):  # Handle spike sequences at the end of the data
            # Replace end spikes with the mean of non-NaN values in the dataset
            modified_data[start:end + 1] = mean(velocities[~np.isnan(velocities)])
        else:
            # Use the mean of valid start and end points as a fallback
            fallback_value = np.nanmean([valid_start, valid_end])
            modified_data[start:end + 1] = fallback_value

    # Round the interpolated values to the specified number of decimal places
    modified_data = np.around(modified_data, decimals=decimals)
    return modified_data
