from proadv.statistics.descriptive import mean
import numpy as np


def _random_index(data_size, percent):
    """
    Generate random indexes based on the percentage of data size.

    Parameters
    ------
    ndata (int): The size of the data.
    percent (float): The percentage of pollution to generate.

    Returns
    ------
    randoms (np.ndarray): An array containing randomly selected indexes.

    """

    randoms = []
    for i in range(int(data_size * percent)):
        rn = np.random.randint(0, data_size)  # random number
        if rn in randoms:
            rn = np.random.randint(0, data_size)
            randoms.append(rn)
        else:
            randoms.append(rn)
    return np.array(sorted(randoms))  # Random Indexes


def synthetic_noise(data, percent):
    synthetic_data = _random_index(data.size, percent)

    r = np.random.normal(0, 0.05, data.size)  # Nosie Vector

    s = np.zeros(data.size)  # Artificial Spike

    for k in range(data.size):
        if k % 2 == 0:
            s[synthetic_data[k]] = (abs(r[k]) + 0.8) * mean(data)
        else:
            s[synthetic_data[k]] = (-abs(r[k]) - 0.8) * mean(data)

    synthetic_polluted_data = data + synthetic_data
    return synthetic_polluted_data