# Kernel Density Estimation-based Despiking Algorithm (KDE)

The Kernel Density Estimation (KDE) algorithm provides a non-iterative method for despiking, based on the kernel bivariate distribution function. It calculates a 2D density estimate of the dataset obtained from ADV measurements.

In KDE, the dataset is normalized and centered, and then the density is computed using kernel equations. This estimation is based on analyzing the correlation between velocities in different directions. 

By calculating the kernel density estimation of *u-Δu*, KDE identifies peaks, from which density profiles are extracted. 

Using these profiles, the algorithm determines cutoff points and a criteria ellipse. Points outside this ellipse are considered spikes. The algorithm was developed to handle datasets with significant contamination, exceeding 40%.
