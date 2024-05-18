from proadv.statistics.descriptive import mean
import numpy as np


def _random_index(ndata, percent):
    r = []
    for i in range(int(ndata * percent)):
        rn = np.random.randint(0, ndata)  # random number
        if rn in r:
            rn = np.random.randint(0, ndata)
            r.append(rn)
        else:
            r.append(rn)
    return np.array(sorted(r))  # Random Indexes


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
