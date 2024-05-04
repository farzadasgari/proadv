import numpy as np


def calculate_derivatives(c):
    dc = np.zeros_like(c)
    dc2 = np.zeros_like(c)
    for i in range(1, len(c) - 1):
        dc[i] = np.around((c[i + 1] - c[i - 1]) / 2, 4)
    for i in range(1, len(c) - 1):
        dc2[i] = np.around((dc[i + 1] - dc[i - 1]) / 2, 4)
    return dc, dc2
