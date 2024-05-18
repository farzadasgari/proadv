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
