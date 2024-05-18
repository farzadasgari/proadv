# Kernel Density Estimation-based Despiking Algorithm (KDE)

The Kernel Density Estimation (KDE) algorithm provides a non-iterative method for despiking, based on the kernel bivariate distribution function. It calculates a 2D density estimate of the dataset obtained from ADV measurements.

In KDE, the dataset is normalized and centered, and then the density is computed using kernel equations. This estimation is based on analyzing the correlation between velocities in different directions. 

By calculating the kernel density estimation of *u-Δu*, KDE identifies peaks, from which density profiles are extracted. 

Using these profiles, the algorithm determines cutoff points and a criteria ellipse. Points outside this ellipse are considered spikes. The algorithm was developed to handle datasets with significant contamination, exceeding 40%.

## Kernel Density Estimation (KDE)
Kernel Density Estimation (KDE) is a robust statistical method used to estimate the density of a probability distribution, especially when the underlying model is unknown. KDE involves placing a kernel function at each data point and summing their contributions to obtain a continuous estimate of the Probability Density Function (PDF). The choice of kernel function and bandwidth parameter significantly influences the accuracy of the estimate.

## Applications of KDE
KDE finds applications in various domains including water engineering, such as daily precipitation resampling, particle density estimation, particle tracking simulations, turbulence statistics analysis, and acoustic Doppler data despiking.

## Limitations of Naive KDE
Naive KDE, which assigns equal weights to all data points, is often inadequate. To improve accuracy, an appropriate kernel function is needed. For instance, Gaussian kernel estimation is commonly used in applications like Acoustic Doppler Velocimetry (ADV) despiking.

## Usage

The `bivariate_kernel` function provided by ProADV implements the Bivariate Kernel Density method for spike detection and replacement in velocity data.

```python
from proadv.filtration.detection.bivariatekernel import _cutoff, _derivative
from proadv.kernel.bivariate import _rotation
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde


def main():
    """
    Example function demonstrating the use of bivariate kernel density detection for data filtration.

    Reads a CSV file containing velocity data, applies bivariate kernel density detection iteratively,
    and plots the filtered results.
    """

    # Read velocity data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values

    # Calculate derivative of the velocity data
    derivative = _derivative(main_data)

    # Determine rotation angle for transformation
    theta = _rotation(main_data, derivative)

    # Transform velocity data
    ut = main_data * np.cos(theta) + derivative * np.sin(theta)
    wt = -main_data * np.sin(theta) + derivative * np.cos(theta)

    # Generate grid for kernel density estimation
    grid = 256
    xmin = ut.min()
    xmax = ut.max()
    ymin = wt.min()
    ymax = wt.max()
    X, Y = np.mgrid[xmin:xmax:grid * 1j, ymin:ymax:grid * 1j]
    uf = X[:, 0]
    wf = Y[0, :]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([ut, wt])

    # Perform kernel density estimation
    kernel = kde(values)
    density = np.reshape(kernel(positions).T, X.shape)
    dp = density.max()
    up, wp = np.where(density == dp)[0][0], np.where(density == dp)[1][0]
    fu = density[:, wp]
    fw = density[up, :]

    # Determine cutoff values
    ul, uu = _cutoff(dp, uf, 0.4, 0.4, fu, up, grid)
    wl, wu = _cutoff(dp, wf, 0.4, 0.4, fw, wp, grid)
    uu1 = uu - 0.5 * (uu + ul)
    ul1 = ul - 0.5 * (uu + ul)
    wu1 = wu - 0.5 * (wu + wl)
    wl1 = wl - 0.5 * (wu + wl)
    Ut1 = ut - 0.5 * (uu + ul)
    Wt1 = wt - 0.5 * (wu + wl)

    # Initialize array for spike detection
    F = np.zeros(main_data.size)
    at = 0.5 * (uu1 - ul1)
    bt = 0.5 * (wu1 - wl1)

    # Perform spike detection
    for i in range(0, main_data.size):
        if Ut1[i] > uu1 or Ut1[i] < ul1:
            F[i] = 1
        else:
            we = np.sqrt((bt ** 2) * (1 - (Ut1[i] ** 2) / (at ** 2)))
            if Wt1[i] > we or Wt1[i] < -we:
                F[i] = 1

    # Identify spike indices
    indices = np.where(F > 0)[0]

    # Perform data interpolation for spike removal
    filtered_data = linear_interpolation(main_data, indices)

    # Plotting the filtered and unfiltered data
    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data.size * delta_time, delta_time)
    plt.plot(time, main_data, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data, color='black', label='Filtered')
    plt.title('Bivariate Kernel Density')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
```

![bivariate-kernel](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/bivariate-kernel.png)

This example demonstrates the use of the `bivariate kernel` function to iteratively filter velocity data stored in a CSV file. The function iterates through the data multiple times, detecting and replacing spikes using phase-space thresholding. The filtered data is then plotted alongside the original data for visualization.

---

**Using Botev Bivariate Kernel Function:**

For improved accuracy and robustness in bivariate kernel density estimation, the `proadv.kernel.bivariate.bivariate_kernel` function can be employed. This function leverages advanced algorithms, such as those proposed by [Botev et al. (2010)](https://projecteuclid.org/journals/annals-of-statistics/volume-38/issue-5/Kernel-density-estimation-via-diffusion/10.1214/10-AOS799.full), to calculate the kernel density with enhanced precision and reliability. By utilizing state-of-the-art methodologies, researchers can achieve more accurate results, especially in challenging datasets with complex velocity distributions and high levels of noise or spikes.

## Bivariate Kernel Density Estimation

## Introduction

Bivariate kernel density estimation is a statistical method used to estimate the probability density function of a two-dimensional random variable. It extends the concept of univariate kernel density estimation to higher dimensions, allowing for the visualization and analysis of the joint distribution of two variables.

The Botev method, introduced by Zdravko Botev, is a specific approach to bivariate kernel density estimation that aims to improve accuracy and computational efficiency compared to other methods.

## Botev Method Overview

The Botev method for bivariate kernel density estimation involves the following key steps:

1. **Data Preparation:** Begin by preparing your data, which consists of pairs of observations from the two variables of interest.

2. **Bandwidth Selection:** Choose an appropriate bandwidth parameter, which determines the smoothness of the estimated density. The Botev method employs a data-driven approach to automatically select an optimal bandwidth based on the input data.

3. **Kernel Function:** Select a kernel function, which determines the shape of the kernel used to smooth the data. Common choices include Gaussian, Epanechnikov, and uniform kernels.

4. **Density Estimation:** Use the selected kernel function and bandwidth to estimate the density at each point in the two-dimensional space defined by the range of the two variables. This involves calculating the kernel density estimate for each observation pair and summing them to obtain the overall density estimate.

5. **Normalization:** Normalize the density estimate to ensure that it integrates to unity over the entire two-dimensional space. This step is essential for obtaining a valid probability density function.

6. **Optional: Visualization:** Visualize the estimated density using contour plots, heatmaps, or other graphical representations to gain insights into the joint distribution of the two variables.

## Conclusion

Bivariate kernel density estimation with the Botev method is a powerful technique for analyzing the joint distribution of two variables. By estimating the probability density function in two dimensions, it provides valuable insights into the relationship between the variables and can be used for various statistical and machine learning. 

The bivariate kernel density detection method exhibits commendable performance in highly polluted velocity datasets, effectively identifying and removing spikes. However, its performance tends to diminish in datasets with lower levels of pollution, where its ability to differentiate spikes from genuine data points may be compromised. Despite its limitations, the bivariate kernel density method remains a valuable tool for data filtration, particularly in scenarios characterized by significant noise or spikes.

## References
[Islam, Md Rashedul, and David Z. Zhu. "Kernel density–based algorithm for despiking ADV data." Journal of Hydraulic Engineering 139, no. 7 (2013): 785-793.](https://doi.org/10.1061/(ASCE)HY.1943-7900.0000734)

[Botev, Zdravko I., Joseph F. Grotowski, and Dirk P. Kroese. "Kernel density estimation via diffusion." (2010): 2916-2957.](https://projecteuclid.org/journals/annals-of-statistics/volume-38/issue-5/Kernel-density-estimation-via-diffusion/10.1214/10-AOS799.full)