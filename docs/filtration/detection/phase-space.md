# Phase-Space Thresholding (PST)

The Phase-Space Thresholding (PST) method stands as one of the fundamental spike detection techniques, serving as a cornerstone for various subsequent algorithms. Introduced as a robust approach in signal processing, it operates on phase-space equations of velocity and its derivatives (*u−Δu−Δ<sup>2</sup>u*), allowing for efficient despiking operations.

Spatial equations form the backbone of PST calculations, particularly with the criteria ellipse, derived from data acquired through ADV measurements. This elliptical representation captures the statistical features of velocity data, facilitating the identification and replacement of spikes in the signal.

## Algorithm

Given the complexities of 3D space calculations, PST computations are often simplified to 2D space, leveraging standard mathematical techniques for spike detection and replacement. By comparing data plots (*u−Δu*, *u−Δ<sup>2</sup>u*, and *Δu−Δ<sup>2</sup>u*) with criteria ellipses, spikes lying outside the ellipse boundaries are pinpointed and replaced across all velocity components (u, v, and w).

While PST is predominantly applied in 2D space, efforts have been made to extend its applicability to 3D scenarios. The iterative nature of the PST algorithm ensures comprehensive coverage, aiming to keep all data points within the criteria ellipse boundaries.

## Usage

The `phasespace` function provided by ProADV implements the Phase-Space Thresholding method for spike detection and replacement in velocity data.

```python
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
```


![phase-space](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/phase-space.png)

This example demonstrates the use of the `phasespace` function to iteratively filter velocity data stored in a CSV file. The function iterates through the data multiple times, detecting and replacing spikes using phase-space thresholding. The filtered data is then plotted alongside the original data for visualization.

## Conclusion
Overall, the PST method represents a robust approach to spike detection and replacement, with ongoing research aiming to refine its applicability across diverse datasets and dimensions.

## References
[Goring, Derek G., and Vladimir I. Nikora. "Despiking acoustic Doppler velocimeter data." Journal of hydraulic engineering 128, no. 1 (2002): 117-126.](https://doi.org/10.1061/(ASCE)0733-9429(2002)128:1(117))
