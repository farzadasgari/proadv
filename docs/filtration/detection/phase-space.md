# Phase-Space Thresholding (PST)

The Phase-Space Thresholding (PST) method stands as one of the fundamental spike detection techniques, serving as a cornerstone for various subsequent algorithms. Introduced as a robust approach in signal processing, it operates on phase-space equations of velocity and its derivatives (*u−Δu−Δ<sup>2</sup>u*), allowing for efficient despiking operations.

Spatial equations form the backbone of PST calculations, particularly with the criteria ellipse, derived from data acquired through ADV measurements. This elliptical representation captures the statistical features of velocity data, facilitating the identification and replacement of spikes in the signal.

Given the complexities of 3D space calculations, PST computations are often simplified to 2D space, leveraging standard mathematical techniques for spike detection and replacement. By comparing data plots (*u−Δu*, *u−Δ<sup>2</sup>u*, and *Δu−Δ<sup>2</sup>u*) with criteria ellipses, spikes lying outside the ellipse boundaries are pinpointed and replaced across all velocity components (u, v, and w).

While PST is predominantly applied in 2D space, efforts have been made to extend its applicability to 3D scenarios. The iterative nature of the PST algorithm ensures comprehensive coverage, aiming to keep all data points within the criteria ellipse boundaries.

Overall, the PST method represents a robust approach to spike detection and replacement, with ongoing research aiming to refine its applicability across diverse datasets and dimensions.
