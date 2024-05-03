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


def linear_interpolation(velocities, spike_indices):
    from proadv.statistics.desc import mean
    modified_data = np.copy(velocities)
    modified_data[spike_indices] = np.nan
    for idx in spike_indices:
        try:
            modified_data[idx] = np.around(
                mean(modified_data[idx - 1] + modified_data[idx:][~np.isnan(modified_data[idx:])][0]),
                decimals=4)
        except IndexError:
            modified_data[idx] = mean(velocities)
    return modified_data
