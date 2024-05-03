import numpy as np


def acceleration_thresholding(velocities, frequency, tag, gravity=980, k_gravity=1.5, k_sigma=1):
    velocities = np.asarray(velocities)
    differences = np.concatenate((np.array([0]), np.diff(velocities) * frequency))
    print(differences, len(differences))
    if tag == 1:
        accel_indices = np.intersect1d(np.where(differences > k_gravity * gravity)[0],
                                       np.where(velocities > np.mean(velocities) + k_sigma * np.std(velocities))[0])
    elif tag == 2:
        accel_indices = np.intersect1d(np.where(differences < -k_gravity * gravity)[0],
                                       np.where(velocities < np.mean(velocities) - k_sigma * np.std(velocities))[0])
    else:
        raise ValueError("Invalid tag value. Expected 1 or 2.")

    return accel_indices
