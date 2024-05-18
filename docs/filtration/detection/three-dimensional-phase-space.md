# Three Dimesional Phase-Space Thresholding (3d-PST)

While Phase-Space Thresholding (PST) is traditionally applied in 2D space, recent efforts have been directed towards extending its applicability to 3D scenarios. This expansion aims to accommodate the complexities of three-dimensional data analysis, allowing for a more comprehensive assessment of velocity fluctuations across multiple axes.

The iterative nature of the PST algorithm remains a cornerstone of its functionality, ensuring thorough coverage in identifying and addressing spikes within the dataset. By iteratively refining the criteria ellipse boundaries in three-dimensional space, PST aims to encompass all data points effectively while maintaining the integrity of the underlying statistical features.

This adaptation underscores PST's adaptability and versatility, enabling its usage across diverse datasets and dimensions. The ongoing refinement of PST for 3D applications reflects a commitment to enhancing its efficacy in addressing complex velocity fluctuations and spike detection in three-dimensional velocity datasets.

## Usage

The `spherical_phasespace_thresholding` function provided by ProADV implements the Phase-Space Thresholding method for spike detection and replacement in velocity data.

```python
import proadv as adv
from proadv.filtration.detection.spherical import spherical_phasespace_thresholding
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    """
    Example function demonstrating the use of three-dimensional phase-space thresholding
        and linear interpolation for data filtration.

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
    max_iteration = 3

    while iteration < max_iteration:

        # Calculate average value for the current filtered data
        average_x = adv.statistics.descriptive.mean(filtered_data_x)
        average_y = adv.statistics.descriptive.mean(filtered_data_y)
        average_z = adv.statistics.descriptive.mean(filtered_data_z)

        # Apply phase-space thresholding to detect outliers
        indices_x = spherical_phasespace_thresholding(filtered_data_x - average_x, iteration, average_x)
        indices_y = spherical_phasespace_thresholding(filtered_data_y - average_y, iteration, average_y)
        indices_z = spherical_phasespace_thresholding(filtered_data_z - average_z, iteration, average_z)

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
    plt.title('Three-Dimensional Phase-Space Thresholding')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
```


![3d-phase-space](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/phase-space-3d.png)

This example demonstrates the use of the `spherical_phasespace_thresholding` function to iteratively filter velocity data stored in a CSV file. The function iterates through the data multiple times, detecting and replacing spikes using three-dimesional phase-space thresholding. The filtered data is then plotted alongside the original data for visualization.

## References
[Mori, Nobuhito, Takuma Suzuki, and Shohachi Kakuno. "Noise of acoustic Doppler velocimeter data in bubbly flows." Journal of engineering mechanics 133, no. 1 (2007): 122-125.](https://doi.org/10.1061/(ASCE)0733-9399(2007)133:1(12))
