# Kernel Density Estimation (KDE)
Kernel Density Estimation (KDE) is a robust statistical method used to estimate the density of a probability distribution, especially when the underlying model is unknown. KDE involves placing a kernel function at each data point and summing their contributions to obtain a continuous estimate of the Probability Density Function (PDF). The choice of kernel function and bandwidth parameter significantly influences the accuracy of the estimate.

## Applications of KDE
KDE finds applications in various domains including water engineering, such as daily precipitation resampling, particle density estimation, particle tracking simulations, turbulence statistics analysis, and acoustic Doppler data despiking.

## Limitations of Naive KDE
Naive KDE, which assigns equal weights to all data points, is often inadequate. To improve accuracy, an appropriate kernel function is needed. For instance, Gaussian kernel estimation is commonly used in applications like Acoustic Doppler Velocimetry (ADV) despiking.
