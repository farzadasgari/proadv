import proadv as adv
from proadv.filtration.detection.phasespace import phasespace_thresholding
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values
    filtered_data = main_data.copy()
    iteration = 0
    max_iteration = 3
    while iteration < max_iteration:
        indices = phasespace_thresholding(filtered_data - adv.statistics.descriptive.mean(filtered_data))
        if not indices.size:
            break
        filtered_data = linear_interpolation(filtered_data, indices, decimals=2)
        iteration += 1

    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data.size * delta_time, delta_time)
    plt.plot(time, main_data, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data, color='black', label='Filtered')
    plt.title('Phase-Space Thresholding')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
