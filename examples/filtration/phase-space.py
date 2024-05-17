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
    main_data = df.iloc[:, 1].values
    filtered_data = main_data.copy()

    # Iterative filtration process
    iteration = 0
    max_iteration = 10

    while iteration < max_iteration:

        # Calculate average value for the current filtered data
        average = adv.statistics.descriptive.mean(filtered_data)

        # Apply phase-space thresholding to detect outliers
        indices = phasespace_thresholding(filtered_data - average)

        # Break loop if no outliers are detected
        if not indices.size:
            break

        # Replace outliers with interpolated values
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
