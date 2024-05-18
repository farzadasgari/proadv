import proadv as adv
from proadv.filtration.detection.trivariatekernel import three_dimensional_kernel
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    # Read velocity data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data_x = df.iloc[:, 0].values
    main_data_y = df.iloc[:, 1].values
    main_data_z = df.iloc[:, 2].values

    indices_x = three_dimensional_kernel(main_data_x, 64)
    indices_y = three_dimensional_kernel(main_data_y, 64)
    indices_z = three_dimensional_kernel(main_data_z, 64)

    indices = np.sort(np.unique(np.concatenate((indices_x, indices_y, indices_z))))

    filtered_data_x = linear_interpolation(main_data_x, indices, decimals=3)
    filtered_data_y = linear_interpolation(main_data_y, indices, decimals=3)
    filtered_data_z = linear_interpolation(main_data_z, indices, decimals=3)

    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data_y.size * delta_time, delta_time)
    plt.plot(time, main_data_y, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data_y, color='black', label='Filtered')
    plt.title('Three-Dimensional Kernel Density')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
