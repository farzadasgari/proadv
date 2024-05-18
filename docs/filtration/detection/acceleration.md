# Acceleration Thresholding (AT)

The Acceleration Thresholding (AT) method is a traditional approach to spike detection and replacement, characterized by two distinct phases: one targeting negative acceleration and the other positive accelerations. 

In each phase, the algorithm iterates through the data multiple times, examining each data point to ensure compliance with the acceleration criterion and magnitude threshold. This iterative process guarantees thorough spike detection and replacement.

While the AT method is fast and effective for detecting and replacing simple spikes, it may not perform well with complex spikes that exhibit non-linear behavior or unusual patterns.

## Algorithm 

This method employs a detection and replacement approach involving two distinct phases: one targeting negative accelerations and the other positive accelerations. Within each phase, the algorithm iterates through the dataset multiple times until all data points adhere to the acceleration criterion and the magnitude threshold.

## Usage

The `acceleration_thresholding` function provided by ProADV implements the Acceleration Thresholding method for spike detection and replacement in velocity data.

```python

import proadv as adv
from proadv.filtration.detection.acceleration import acceleration_thresholding
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    """
    Example function demonstrating the use of acceleration thresholding and linear interpolation for data filtration.

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
    tag = 1

    while iteration < max_iteration:

        # Toggle tag between 1 and 2 for acceleration thresholding
        tag = 2 if tag == 1 else 1

        # Calculate average value for the current filtered data
        average_x = adv.statistics.descriptive.mean(filtered_data_x)
        average_y = adv.statistics.descriptive.mean(filtered_data_y)
        average_z = adv.statistics.descriptive.mean(filtered_data_z)

        # Apply acceleration thresholding to detect outliers
        indices_x = acceleration_thresholding(filtered_data_x - average_x, frequency=100, tag=tag,
                                              gravity=980, k_gravity=1.5, k_sigma=1)
        indices_y = acceleration_thresholding(filtered_data_y - average_y, frequency=100, tag=tag,
                                              gravity=980, k_gravity=1.5, k_sigma=1)
        indices_z = acceleration_thresholding(filtered_data_z - average_z, frequency=100, tag=tag,
                                              gravity=980, k_gravity=1.5, k_sigma=1)

        indices = np.sort(np.unique(np.concatenate((indices_x, indices_y, indices_z))))

        # Break loop if no outliers are detected
        if not indices.size:
            break

        # Replace outliers with interpolated values
        filtered_data_x = linear_interpolation(filtered_data_x, indices, decimals=3)
        filtered_data_y = linear_interpolation(filtered_data_y, indices, decimals=3)
        filtered_data_z = linear_interpolation(filtered_data_z, indices, decimals=3)

        iteration += 1

    # Plotting the filtered and unfiltered data

    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data_x.size * delta_time, delta_time)
    plt.plot(time, main_data_x, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data_x, color='black', label='Filtered')
    plt.title('Acceleration Thresholding')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
```


![acceleration](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/acceleration.png)

This example demonstrates the use of the `acceleration_thresholding` function to iteratively filter velocity data stored in a CSV file. The function iterates through the data multiple times, detecting and replacing spikes using acceleration thresholding. The filtered data is then plotted alongside the original data for visualization.

## References
[Goring, Derek G., and Vladimir I. Nikora. "Despiking acoustic Doppler velocimeter data." Journal of hydraulic engineering 128, no. 1 (2002): 117-126.](https://doi.org/10.1061/(ASCE)0733-9429(2002)128:1(117))
