import numpy as np


def last_valid_data(velocities, spike_indices):
    """
    Replace spike values in velocities array with the last valid data before each.

    Parameters:
        velocities (numpynd.array): Array of velocity data.
            An array-like object containing velocity values.
        spike_indices (numpy.ndarray): Indices of spikes.
            An array-like object containing the indices of detected spikes.

    Returns:
        modified_data (numpy.ndarray): Modified data with spikes replaced by last valid values.
            An array containing the modified data.
    """
    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Replace values at spikes indices with the last valid value
    for idx in spike_indices:
        modified_data[idx] = modified_data[idx - 1]

    return modified_data


def linear_interpolation(velocities, spike_indices, decimals=4):
    """
    Replace spike values in velocity data with linearly interpolated values
        between the nearest valid data points before and after each spike.

    Parameters
    ------
        velocities (numpy.ndarray): Array of velocity data.
            An array-like object containing velocity values.
        spike_indices (numpy.ndarray): Indices of spike events.
            An array-like object containing the indices of detected spike events.
        decimals (int): Number of decimal places to round the interpolated values.
            Default is 4.

    Returns
    ------
        modified_data (numpy.ndarray): Modified velocity data with spikes replaced by interpolated values.
            An array containing the modified velocity data.
    """
    from proadv.statistics.descriptive import mean

    # Create a copy of velocity data
    modified_data = np.copy(velocities)

    # Replace spike values with linearly interpolated values
    modified_data[spike_indices] = np.nan

    for idx in spike_indices:
        # Linear interpolation
        try:
            modified_data[idx] = mean((modified_data[idx - 1], modified_data[idx:][~np.isnan(modified_data[idx:])][0]))
        except IndexError:
            modified_data[idx] = mean(velocities)

    # Replace spikes without valid data in one side.
    # Like first and last data points.
    modified_data[np.isnan(modified_data)] = mean(velocities)

    return np.around(modified_data, decimals=decimals)
