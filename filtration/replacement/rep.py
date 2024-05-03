import numpy as np


def last_valid_data(velocities, spike_indices):
    modified_data = np.copy(velocities)
    for idx in spike_indices:
        modified_data[idx] = modified_data[idx - 1]
    return modified_data
