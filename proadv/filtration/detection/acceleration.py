import numpy as np
from proadv.statistics.spread import std


def acceleration_thresholding(velocities, frequency, tag, gravity=980, k_gravity=1.5, k_sigma=1):
    """
    Detects acceleration events based on velocity data.

    Calculate acceleration events based on velocity data, considering thresholds
        for acceleration magnitude and velocity deviation from mean.

    Parameters
    ------
        velocities (array_like): Array of velocity data.
            An array-like object containing velocity values.
        frequency (float): Sampling frequency.
            The frequency at which the velocity data is sampled,
            used to calculate acceleration from velocity differences.
        tag (int): Tag for acceleration direction.
            An integer indicating the direction of acceleration to detect:
                - 1 for positive acceleration (increasing velocity)
                - 2 for negative acceleration (decreasing velocity).
        gravity (float): Value of gravity.
            The acceleration due to gravity, used as a reference for acceleration thresholds.
            Default is 980 (m/s^2).
        k_gravity (float): Threshold multiplier for gravity.
            A multiplier applied to the gravity value to determine the acceleration threshold.
            Default is 1.5.
        k_sigma (float, optional): Threshold multiplier for standard deviation.
            A multiplier applied to the standard deviation of velocities to determine velocity deviation threshold.
            Default is 1.

    Returns
    ------
        accel_indices (array_like): Indices of acceleration events.
            An array containing the indices of the detected acceleration events.

    Raises
    ------
        ValueError: If invalid tag value is provided.
            Raised when the tag value is not 1 or 2.

    References
    ------
        Goring, Derek G., and Vladimir I. Nikora.
            "Despiking acoustic Doppler velocimeter data."
            Journal of hydraulic engineering 128.1 (2002): 117-126.
    """
    velocities = np.asarray(velocities)

    # Calculate velocity differences and multiply by frequency to obtain acceleration
    differences = np.concatenate((np.array([0]), np.diff(velocities) * frequency))

    if tag == 1:
        # Detect positive acceleration events
        accel_indices = np.intersect1d(np.where(differences > k_gravity * gravity)[0],
                                       np.where(velocities > np.mean(velocities) + k_sigma * std(velocities))[0])
    elif tag == 2:
        # Detect negative acceleration events
        accel_indices = np.intersect1d(np.where(differences < -k_gravity * gravity)[0],
                                       np.where(velocities < np.mean(velocities) - k_sigma * std(velocities))[0])
    else:
        raise ValueError("Invalid tag value. Expected 1 or 2.")

    return accel_indices
