import numpy as np


def _descript(c, iteration, c_mean):
    """
    Calculate the mean value and center the input data.

    Parameters
    ------
    c (array_like): Input data.
    iteration (int): Loop counter.
    c_mean (float): Mean value of input data.

    Returns
    ------
    f (array_like): Centered input data.
    f_mean (float): Updated mean value.
    """
    if iteration == 1:
        f_mean = np.around(np.nanmean(c), 4)
    else:
        f_mean = np.around(c_mean + np.nanmean(c), 4)
    f = np.around(c - np.nanmean(c), 4)
    return f, f_mean


def _gradients(c):
    """
    Calculate the first and second order derivatives of the input data.

    Parameters
    ------
    c (array_like): Input data.

    Returns
    ------
    dc (array_like): First order derivative of the input data.
    dc2 (array_like): Second order derivative of the input data.
    """
    dc = np.gradient(c)
    dc2 = np.gradient(dc)
    return dc, dc2


def _rotation(c, dc, dc2, theta):
    """
    Rotate the input data based on the specified angle theta.

    Parameters
    ------
    c (array_like): Input data.
    dc (array_like): First order derivative of the input data.
    dc2 (array_like): Second order derivative of the input data.

    Returns
    ------
    x (array_like): Rotated X-axis data.
    y (array_like): Rotated Y-axis data.
    z (array_like): Rotated Z-axis data.
    """
    if theta == 0:
        x = c.copy()
        y = dc.copy()
        z = dc2.copy()
    else:
        rotation_matrix = np.around(np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]), 4)
        x = np.around(c * rotation_matrix[0, 0] + dc * rotation_matrix[0, 1] + dc2 * rotation_matrix[0, 2], 4)
        y = np.around(c * rotation_matrix[1, 0] + dc * rotation_matrix[1, 1] + dc2 * rotation_matrix[1, 2], 4)
        z = np.around(c * rotation_matrix[2, 0] + dc * rotation_matrix[2, 1] + dc2 * rotation_matrix[2, 2], 4)
    return x, y, z


def _parameters(x, y, z):
    """
    Calculate parameters for phase-space thresholding in spherical coordinates.

    Parameters
    ------
    x (array_like): Rotated X-axis data.
    y (array_like): Rotated Y-axis data.
    z (array_like): Rotated Z-axis data.

    Returns
    ------
    a (float): Coefficient 'a'.
    b (float): Coefficient 'b'.
    c (float): Coefficient 'c'.
    """
    lambda_ = np.around(np.sqrt(2 * np.log(x.size)), 4)
    a = np.around(lambda_ * np.nanstd(x), 4)
    b = np.around(lambda_ * np.nanstd(y), 4)
    c = np.around(lambda_ * np.nanstd(z), 4)
    return a, b, c


def _spike_indices(x, y, z, a, b, c):
    """
    Calculate spike indices based on the input data and coefficients.

    Parameters
    ------
    x (array_like): Rotated X-axis data.
    y (array_like): Rotated Y-axis data.
    z (array_like): Rotated Z-axis data.
    a (float): Coefficient 'a'.
    b (float): Coefficient 'a'.
    c (float): Coefficient 'a'.

    Returns
    ------
    spike_indices (array_like): Indices of detected spikes.
    """
    xp, yp, zp, ip = [], [], [], []
    for i in range(x.size):
        x1 = x[i]
        y1 = y[i]
        z1 = z[i]
        x2 = np.around(a * b * c * x1 / np.sqrt((a * c * y1) ** 2 + b ** 2 * (c ** 2 * x1 ** 2 + a ** 2 * z1 ** 2)), 4)
        y2 = np.around(a * b * c * y1 / np.sqrt((a * c * y1) ** 2 + b ** 2 * (c ** 2 * x1 ** 2 + a ** 2 * z1 ** 2)), 4)
        zt = np.around(c ** 2 * (1 - (x2 / a) ** 2 - (y2 / b) ** 2), 4)
        if z1 < 0:
            z2 = -np.around(np.sqrt(zt), 4)
        elif z1 > 0:
            z2 = np.around(np.sqrt(zt), 4)
        else:
            z2 = 0
        dis = (x2 ** 2 + y2 ** 2 + z2 ** 2) - (x1 ** 2 + y1 ** 2 + z1 ** 2)
        if dis < 0:
            ip.append(i)
            xp.append(x[i])
            yp.append(y[i])
            zp.append(z[i])
    spike_indices = np.array(ip, dtype=np.int64)
    return spike_indices


def spherical_phasespace_thresholding(c, iteration, c_mean):
    """
    Detect spikes using three-dimensional phase-space thresholding, based on each velocity component and
        their first-order and second-order derivatives.

    Parameters
    ------
    c (array_like): Velocity component
    iteration (int): Loop counter.
    c_mean (float): Mean of the velocity component

    Returns
    ------
    spherical_indices (array_like): Array containing the indices of detected spikes.

    References
    ------
        Wahl, Tony L.
        "Discussion of “Despiking acoustic doppler velocimeter data” by Derek G. Goring and Vladimir I. Nikora."
        Journal of Hydraulic Engineering 129, no. 6 (2003): 484-487.
    """

    # Calculate mean and center the data
    c, c_mean = _descript(c, iteration, c_mean)

    # Calculate gradients
    dc, dc2 = _gradients(c)

    # Calculaterotation angle
    theta = np.around(np.arctan2(np.sum(c * dc), np.sum(dc2 ** 2)), 4)

    # Rotate data
    x, y, z = _rotation(c, dc, dc2, theta)

    # Calculate parameters
    a, b, c = _parameters(x, y, z)

    # Calculate spike indices
    spherical_indices = _spike_indices(x, y, z, a, b, c)

    return spherical_indices
