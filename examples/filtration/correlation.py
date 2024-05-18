import proadv as adv
from proadv.filtration.detection.correlation import velocity_correlation
from proadv.filtration.replacement.replacements import mean_value
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    """
    Example function demonstrating the use of velocity correlation filter and overal mean value for data filtration.

    Reads a CSV file containing velocity data, applies filtration methods iteratively, and plots the results.
    """

    # Read velocity data from CSV file
    df = read_csv('../../dataset/first.csv')
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

        # Apply velocity correlation to detect outliers
        indices = velocity_correlation(filtered_data_x - average_x,
                                       filtered_data_y - average_y,
                                       filtered_data_z - average_z)

        # Break loop if no outliers are detected
        if not indices.size:
            break

        # Replace outliers with interpolated values
        filtered_data_x = mean_value(filtered_data_x, indices)
        filtered_data_y = mean_value(filtered_data_y, indices)
        filtered_data_z = mean_value(filtered_data_z, indices)

        iteration += 1

    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data_y.size * delta_time, delta_time)
    plt.plot(time, main_data_y, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data_y, color='black', label='Filtered')
    plt.title('Velocity Correlation')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
