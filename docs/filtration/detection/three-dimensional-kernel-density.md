# Three-Dimesnional Kernel Density Estimation (3d-KDE)

## Trivariate KDE Approach

For 3D kernel density estimation, a trivariate KDE approach is necessary. This method estimates the PDF of velocity by
considering the joint distribution of three variables: velocity, first-order derivative (slope), and second-order
derivative (curvature). The choice of kernel function, such as multivariate normal or Epanechnikov, influences the shape
of the density estimator.

## Algorithm

1. Calculate first-order (slope) and second-order (curvature) derivatives of the velocity time-series signal.
2. Estimate the rotation angle of principal axes and align the three-dimensional matrix with X, Y, and Z axes.
3. Compute trivariate KDE of standardized data.
4. Determine density peak coordinates.
5. Extract density profiles.
6. Calculate rescaled data density slopes to estimate cutoff points.
7. Remove data points outside defined criteria ellipsoid to produce a clean velocity field.

## Usage

The `three_dimensional_kernel` function provided by ProADV implements the Trivariate Kernel Density method for spike
detection and replacement in velocity data.

```python
import proadv as adv
from proadv.filtration.detection.trivariatekernel import three_dimensional_kernel
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def main():
    """
    Main function to demonstrate the application of Three-Dimensional Kernel Density Estimation (3d-KDE)
        for data filtration.

    Reads velocity data from a CSV file, applies trivariate kernel density estimation to identify spikes,
        and performs linear interpolation to filter the data.
    """

    # Read velocity data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data_x = df.iloc[:, 0].values
    main_data_y = df.iloc[:, 1].values
    main_data_z = df.iloc[:, 2].values

    # Calculate spike indices for each velocity component
    indices_x = three_dimensional_kernel(main_data_x, 64)
    indices_y = three_dimensional_kernel(main_data_y, 64)
    indices_z = three_dimensional_kernel(main_data_z, 64)

    # Combine spike indices from all velocity components
    indices = np.sort(np.unique(np.concatenate((indices_x, indices_y, indices_z))))

    # Filter each velocity component using linear interpolation
    filtered_data_x = linear_interpolation(main_data_x, indices, decimals=3)
    filtered_data_y = linear_interpolation(main_data_y, indices, decimals=3)
    filtered_data_z = linear_interpolation(main_data_z, indices, decimals=3)

    # Plot the unfiltered and filtered velocity data
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
```

![trivariate-kernel](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/trivariate-kernel.png)

This example demonstrates the use of the `trivariate kernel` function to iteratively filter velocity data stored in a
CSV file. The function iterates through the data multiple times, detecting and replacing spikes using phase-space
thresholding. The filtered data is then plotted alongside the original data for visualization.

## Conclusion

The Three-Dimensional Kernel Density Estimation (3d-KDE) method demonstrates promising performance in analyzing Doppler
Velocimeter data, particularly in despiking applications. By leveraging the trivariate KDE approach, this method
effectively captures the joint distribution of velocity, slope, and curvature, enabling precise identification and
removal of outliers.

While the 3d-KDE algorithm offers significant advantages in enhancing data quality and reliability, it is important to
note that its computational efficiency can be a limiting factor, especially for large datasets. However, optimizations
such as utilizing Fast Kernel Density Estimation (fastKDE) techniques can significantly alleviate computational burdens
and expedite processing times.

In summary, the 3d-KDE method represents a powerful tool for despiking Doppler Velocimeter data, offering researchers a
robust approach to ensure data integrity and accuracy in various applications.

---

## Using FastKDE for Computational Efficiency

To improve computational efficiency in the Three-Dimensional Kernel Density Estimation (3d-KDE) algorithm, consider
utilizing Fast Kernel Density Estimation (fastKDE) techniques. FastKDE offers significant computational efficiency
gains, reducing processing times from hours to seconds, especially for large datasets. By implementing fastKDE,
researchers can expedite data analysis processes without compromising accuracy or reliability.

## Reference

* [Asgari, Farzad, Seyed Hossein Mohajeri, and Mojtaba Mehraein. "Unleashing the power of three-dimensional kernel density estimation for Doppler Velocimeter data despiking." Measurement 225 (2024): 114053.](https://doi.org/10.1016/j.measurement.2023.114053)

* [O’Brien, Travis A., Karthik Kashinath, Nicholas R. Cavanaugh, William D. Collins, and John P. O’Brien. "A fast and objective multidimensional kernel density estimation method: fastKDE." Computational Statistics & Data Analysis 101 (2016): 148-160.](https://doi.org/10.1016/j.csda.2016.02.014)