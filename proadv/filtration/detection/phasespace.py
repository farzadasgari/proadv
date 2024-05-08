import numpy as np
from proadv.filtration.detection.poincare import calculate_rho
from proadv.statistics.spread import std


def calculate_derivatives(c):
    """
    Calculate time-independent first and second order derivatives of the input data.

    Parameters
    ------
        c (array_like): Input data.

    Returns
    ------
        dc (array_like): First derivative of the input data.
        dc2 (array_like): Second derivative of the input data.
    """
    # Initialize arrays for first and second derivatives
    dc = np.zeros_like(c)
    dc2 = np.zeros_like(c)

    # Calculate first derivative
    for i in range(1, len(c) - 1):
        dc[i] = np.around((c[i + 1] - c[i - 1]) / 2, 4)

    # Calculate second derivative
    for i in range(1, len(c) - 1):
        dc2[i] = np.around((dc[i + 1] - dc[i - 1]) / 2, 4)

    return dc, dc2


def calculate_parameters(c, dc, dc2):
    """
    Calculate parameters for phase-space thresholding.

    Parameters
    ------
        c (array_like): Array of the velocity component.
        dc (array_like): First derivative of the input data.
        dc2 (array_like): Second derivative of the input data.

    Returns
    ------
        std_c (float): Standard deviation of the input data.
        std_dc (float): Standard deviation of the first derivative.
        std_dc2 (float): Standard deviation of the second derivative.
        lambda_ (float): Lambda value.
        theta (float | rad): Angle between components.
        a1 (float): Coefficient 'a1'.
        b1 (float): Coefficient 'b1'.
        a2 (float): Coefficient 'a2'.
        b2 (float): Coefficient 'b2'.
        a3 (float): Coefficient 'a3'.
        b3 (float): Coefficient 'b3'.
    """
    # Calculate standard deviations
    std_c = std(c)
    std_dc = std(dc)
    std_dc2 = std(dc2)

    # Calculate lambda value
    lambda_ = np.sqrt(2 * np.log(len(c)))

    # Calculate theta value
    theta = np.arctan(np.sum(c * dc2) / np.sum(c ** 2))

    # Calculate coefficients
    a1 = lambda_ * std_c
    b1 = lambda_ * std_dc
    a2 = lambda_ * std_dc
    b2 = lambda_ * std_dc2
    fact = np.cos(theta) ** 4 - np.sin(theta) ** 4
    a3 = np.sqrt(a1 ** 2 * np.cos(theta) ** 2 - b2 ** 2 * np.sin(theta) ** 2) / fact
    b3 = np.sqrt(b2 ** 2 * np.cos(theta) ** 2 - a1 ** 2 * np.sin(theta) ** 2) / fact
    return std_c, std_dc, std_dc2, lambda_, theta, a1, b1, a2, b2, a3, b3


def phasespace_thresholding(c):
    """
    Detect spikes using phase-space thresholding, based on each velocity component and
        their first-order and second-order derivatives.

    Phase-space thresholding is a method for detecting spikes or abrupt changes in time series data
        by analyzing the behavior of the data in a multidimensional space defined by the data and its derivatives.


    Parameters
    ------
        c (array_like): Velocity component

    Returns
    ------
        phase_indices (array_like): Array containing the indices of detected spikes.

    References
    ------
        Goring, Derek G., and Vladimir I. Nikora.
            "Despiking acoustic Doppler velocimeter data."
            Journal of hydraulic engineering 128.1 (2002): 117-126.
    """

    # Calculate first and second order derivatives of the input data
    dc, dc2 = calculate_derivatives(c)

    # Calculate parameters used in phase-space thresholding
    std_c, std_dc, std_dc2, lambda_, theta, a1, b1, a2, b2, a3, b3 = calculate_parameters(c, dc, dc2)

    # Calculate poincare map rho values for each dimension
    rho1 = calculate_rho(c, dc, 0, a1, b1)
    rho2 = calculate_rho(dc, dc2, 0, a2, b2)
    rho3 = calculate_rho(c, dc2, theta, a3, b3)

    # Find indices where rho values exceed the threshold
    x1 = np.nonzero(rho1 > 1)[0]
    x2 = np.nonzero(rho2 > 1)[0]
    x3 = np.nonzero(rho3 > 1)[0]

    # Concatenate and sort indices to get unique spike indices
    phase_indices = np.sort(np.unique(np.concatenate((x1, x2, x3))))

    return phase_indices


def phasespace_thresholding_premodification(data, c1=1.5, c2=1.35):
    """
    Perform phase space thresholding premodification on a given data array.

    Parameters
    ------
        data (array_like): Array of data to be processed.
        c1 (float, optional): Threshold parameter for unremovable data. Default is 1.5.
        c2 (float, optional): Threshold parameter for removable data. Default is 1.35.

    Returns
    ------
        data (array_like): Processed data array with potential modifications.
        unremovable_indices (array_like): Indices of unremovable data points.

    Notes
    ------
        This function applies phase space thresholding premodification to the input data array.
        Threshold parameters c1 and c2 determine which data points are considered unremovable and removable, respectively.
        Unremovable data points are those within the range of -c1*theta to c1*theta, where theta is the median absolute deviation.
        Removable data points are those exceeding c2*theta*sqrt(2*log(ndata)), where theta is the median absolute deviation
            and ndata is the length of the data array.
        Removable data points are replaced with the value of the preceding data point.

    Example
    ------
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> phasespace_thresholding_premodification(data)
    """

    data_size = data.size
    theta = np.median(np.absolute(data - np.median(data)))

    # Calculate indices of unremovable data
    unremovable_indices = np.intersect1d(np.nonzero(data >= -c1 * theta)[0], np.nonzero(data <= c1 * theta)[0])

    # Calculate threshold for removable data
    removable_threshold = c2 * theta * np.sqrt(2 * np.log(data_size))

    # Calculate indices of removable data
    removable_indices = np.intersect1d(np.nonzero(data > removable_threshold)[0],
                                       np.nonzero(data < -removable_threshold)[0])

    # Replace removable data points with the preceding data point
    for i in removable_indices:
        data[i] = data[i - 1]

    return data, unremovable_indices
