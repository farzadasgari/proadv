# Three-Dimesnional Kernel Density Estimation (3d-KDE)

## Trivariate KDE Approach
For 3D kernel density estimation, a trivariate KDE approach is necessary. This method estimates the PDF of velocity by considering the joint distribution of three variables: velocity, first-order derivative (slope), and second-order derivative (curvature). The choice of kernel function, such as multivariate normal or Epanechnikov, influences the shape of the density estimator.

## Algorithm Steps
1. Calculate first-order (slope) and second-order (curvature) derivatives of the velocity time-series signal.
2. Estimate the rotation angle of principal axes and align the three-dimensional matrix with X, Y, and Z axes.
3. Compute trivariate KDE of standardized data.
4. Determine density peak coordinates.
5. Extract density profiles.
6. Calculate rescaled data density slopes to estimate cutoff points.
7. Remove data points outside defined criteria ellipsoid to produce a clean velocity field.

## Computational Efficiency
The 3D-KDE algorithm can be computationally intensive. Approaches like converting the trivariate kernel to three two-dimensional kernels or using conditional equations may improve efficiency but require careful implementation. Fast Kernel Density Estimation (fastKDE) offers significant computational efficiency gains, reducing processing time from hours to seconds, especially for large datasets.


## Reference
[Unleashing the power of three-dimensional kernel density estimation for Doppler Velocimeter data despiking](https://doi.org/10.1016/j.measurement.2023.114053)
