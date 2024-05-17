import proadv as adv
from proadv.filtration.detection.phasespace import phasespace_thresholding
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    """
    Example function demonstrating the use of phase-space thresholding and linear interpolation for data filtration.

    Reads a CSV file containing velocity data, applies filtration methods iteratively, and plots the results.
    """

    # Read velocity data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data_x = df.iloc[:, 0].values
    main_data_y = df.iloc[:, 1].values
    main_data_z = df.iloc[:, 2].values
    filtered_data_x = main_data_x.copy()
    filtered_data_y = main_data_y.copy()
    filtered_data_z = main_data_z.copy()

    # Iterative filtration process
    iteration = 0
    max_iteration = 10

    while iteration < max_iteration:

        # Calculate average value for the current filtered data
        average_x = adv.statistics.descriptive.mean(filtered_data_x)
        average_y = adv.statistics.descriptive.mean(filtered_data_y)
        average_z = adv.statistics.descriptive.mean(filtered_data_z)

        # Apply phase-space thresholding to detect outliers
        indices_x = phasespace_thresholding(filtered_data_x - average_x)
        indices_y = phasespace_thresholding(filtered_data_y - average_y)
        indices_z = phasespace_thresholding(filtered_data_z - average_z)

        indices = np.sort(np.unique(np.concatenate((indices_x, indices_y, indices_z))))

        # Break loop if no outliers are detected
        if not indices.size:
            break

        # Replace outliers with interpolated values
        filtered_data_x = linear_interpolation(filtered_data_x, indices, decimals=3)
        filtered_data_y = linear_interpolation(filtered_data_y, indices, decimals=3)
        filtered_data_z = linear_interpolation(filtered_data_z, indices, decimals=3)

        iteration += 1

    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data_x.size * delta_time, delta_time)
    plt.plot(time, main_data_x, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data_x, color='black', label='Filtered')
    plt.title('Phase-Space Thresholding')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
