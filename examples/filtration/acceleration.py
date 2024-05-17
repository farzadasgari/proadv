from proadv.filtration.detection.acceleration import acceleration_thresholding
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values
    filtered_data = main_data.copy()
    iteration = 0
    max_iteration = 10
    tag = 1

    while iteration < max_iteration:
        tag = 2 if tag == 1 else 1
        indices = acceleration_thresholding(filtered_data, frequency=100, tag=tag, gravity=980, k_gravity=1.5,
                                            k_sigma=1)
        if not indices.size:
            break
        filtered_data = linear_interpolation(filtered_data, indices)

        iteration += 1
    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data.size * delta_time, delta_time)
    plt.plot(time, main_data, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data, color='black', label='Filtered')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
