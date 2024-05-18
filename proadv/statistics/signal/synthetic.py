from proadv.statistics.descriptive import mean
import numpy as np


def _random_index(data_size, percent):
    """
    Generate random indexes based on the percentage of data size.

    Parameters
    ------
    ndata (int): The size of the data.
    percent (float): The percentage of artificial pollution to generate.

    Returns
    ------
    randoms (np.ndarray): An array containing randomly selected indexes.
    """

    randoms = []
    iteration = 0
    while iteration < len(range(int(data_size * percent / 100))):
        rn = np.random.randint(0, data_size)  # random number
        if rn in randoms:
            continue
        else:
            randoms.append(rn)
            iteration += 1
    return np.array(sorted(randoms))  # Random Indexes


def synthetic_noise(data, percent):
    """
    Generate synthetic noisy data based on the input data.

    Parameters
    ------
    data (array_like): The original data.
    percent (float): The percentage of data points to perturb.

    Returns
    ------
    synthetic_polluted_data (np.ndarray): Synthetic data with added noise.
    """

    synthetic_data = _random_index(data.size, percent)

    r = np.random.normal(0, 0.05, data.size)  # Nosie Vector

    s = np.zeros(data.size)  # Artificial Spike

    for k in range(synthetic_data.size):
        if k % 2 == 0:
            s[synthetic_data[k]] = (abs(r[k]) + percent) * mean(data)
        else:
            s[synthetic_data[k]] = (-abs(r[k]) - percent) * mean(data)

    synthetic_polluted_data = data + s
    return synthetic_polluted_data
