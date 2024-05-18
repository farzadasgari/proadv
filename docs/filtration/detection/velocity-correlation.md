# Velocity Correlation (VC)

The Velocity Correlation (VC) filter was proposed as an extension of the PST method, aiming to account for the correlation between velocities in different directions. This adaptation addresses challenges encountered in turbulent flows with high concentrations of air bubbles, where conventional replacement algorithms may inadvertently introduce additional spikes.

Unlike the PST method, which analyzes velocities against derivatives, the VCF method plots velocities in all three directions (*u−v−w*) against each other. This approach provides a comprehensive understanding of velocity correlations and facilitates spike detection.

Similar to PST, the criteria ellipse diameters in VCF are determined based on the Universal Threshold and the standard deviation of each velocity component.

## Algorithm
The VCF method treats all three velocity components together during the filtering process. Plotting the three fluctuation velocities against each other enables the inclusion of information from all three velocity components simultaneously. Unlike other algorithms that filter each velocity component independently, the VCF method ensures that if a spike is identified in one velocity component, the other two components are also filtered accordingly.

The filtering criterion used in the VCF method is completely independent of the measuring frequency. Each measurement is treated as a single event, irrespective of contiguous data. Unlike acceleration and phase-space filters, where spike identification depends on the relationship between consecutive measurements, the VCF method employs a criterion based on the relation with the entire sample.

The primary advantage of using a frequency-independent criterion is its ability to handle records with low sampling rates and groups of spikes effectively. However, it forfeits the property exploited by acceleration and phase-space filters, where signal differentiation enhances high-frequency components.

## Usage

The `velocity_correlation` function provided by ProADV implements the Velocity Correlation method for spike detection and replacement in velocity data.

```python
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
```


![correlation](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/correlation.png)

This example demonstrates the use of the `velocity_correlation` function to iteratively filter velocity data stored in a CSV file. The function iterates through the data multiple times, detecting and replacing spikes using velocity_correlation filter. The filtered data is then plotted alongside the original data for visualization.

## References
[Cea, L., J. Puertas, and L. Pena. "Velocity measurements on highly turbulent free surface flow using ADV." Experiments in fluids 42 (2007): 333-348.](https://doi.org/10.1007/s00348-006-0237-3)
