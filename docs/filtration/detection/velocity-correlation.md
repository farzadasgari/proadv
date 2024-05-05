# Velocity Correlation (VC)

The Velocity Correlation (VC) filter was proposed as an extension of the PST method, aiming to account for the correlation between velocities in different directions. This adaptation addresses challenges encountered in turbulent flows with high concentrations of air bubbles, where conventional replacement algorithms may inadvertently introduce additional spikes.

Unlike the PST method, which analyzes velocities against derivatives, the VCF method plots velocities in all three directions (*u−v−w*) against each other. This approach provides a comprehensive understanding of velocity correlations and facilitates spike detection.

Similar to PST, the criteria ellipse diameters in VCF are determined based on the Universal Threshold and the standard deviation of each velocity component.
