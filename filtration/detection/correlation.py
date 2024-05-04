import numpy as np


def calculate_parameters(up, vp, wp):
    """
    Calculate parameters required for velocity correlation.

    Parameters
    ------
        up (numpy.ndarray): Array of the first velocity component.
        vp (numpy.ndarray): Array of the second velocity component.
        wp (numpy.ndarray): Array of the third velocity component.

    Returns
    ------
        lambda_ (float): Lambda value used in velocity correlation detection.
        std_u (float): Standard deviation of the longitudinal velocity component.
        std_v (float): Standard deviation of the transverse velocity component.
        std_w (float): Standard deviation of the vertical velocity component.
    """
    from proadv.statistics.spread import std
    data_size = up.size
    std_u = std(up)
    std_v = std(vp)
    std_w = std(wp)
    lambda_ = np.sqrt(2 * np.log(data_size))
    return lambda_, std_u, std_v, std_w


def calculate_ab(std1, std2, theta, lambda_):
    """
    Calculate 'a' and 'b' coefficients for velocity correlation detection.

    Parameters
    ------
        std1 (float): Standard deviation of the first velocity component.
        std2 (float): Standard deviation of the second velocity component.
        theta (float): Angle between velocity components.
        lambda_: Lambda value used in velocity correlation.

    Returns
    ------
        float: Coefficient 'a' used in velocity correlation.
        float: Coefficient 'b' used in velocity correlation.
    """
    r1 = lambda_ * std1
    r2 = lambda_ * std2
    fact = np.cos(theta) ** 4 - np.sin(theta) ** 4
    fa = (r1 ** 2 * np.cos(theta) ** 2 - r2 ** 2 * np.sin(theta) ** 2) / fact
    fb = (r2 ** 2 * np.cos(theta) ** 2 - r1 ** 2 * np.sin(theta) ** 2) / fact
    return np.sqrt(fa), np.sqrt(fb) if fa > 0 and fb > 0 else (1e10, 1e10)


def calculate_rho(x, y, theta, a, b):
    """
    Calculate rho value for velocity correlation.

    Parameters
    ------
        x (numpy.ndarray): First velocity component.
        y (numpy.ndarray): Second velocity component.
        theta (float): Angle between velocity components.
        a (float): Coefficient 'a' used in velocity correlation.
        b (float): Coefficient 'b' used in velocity correlation.

    Returns
    ------
        rho (numpy.ndarray): Rho value calculated for velocity correlation.
    """
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = y * np.cos(theta) - x * np.sin(theta)
    return (xp / a) ** 2 + (yp / b) ** 2


def velocity_correlation(ui, vi, wi):
    lambda_, std_u, std_v, std_w = calculate_parameters(ui, vi, wi)

    # Calculate angles between velocity components
    theta1 = np.arctan(np.sum(ui * vi) / np.sum(ui ** 2))
    theta2 = np.arctan(np.sum(ui * wi) / np.sum(ui ** 2))
    theta3 = np.arctan(np.sum(vi * wi) / np.sum(vi ** 2))

    # Calculate 'a' and 'b' coefficients for each angle
    a1, b1 = calculate_ab(std_u, std_v, theta1, lambda_)
    a2, b2 = calculate_ab(std_u, std_w, theta2, lambda_)
    a3, b3 = calculate_ab(std_v, std_w, theta3, lambda_)

    # Calculate rho values for each component pair
    rho1 = calculate_rho(ui, vi, theta1, a1, b1)
    rho2 = calculate_rho(ui, wi, theta2, a2, b2)
    rho3 = calculate_rho(vi, wi, theta3, a3, b3)

    # Find indices where rho values exceed 1 (indicating correlation)
    x1 = np.nonzero(rho1 > 1)[0]
    x2 = np.nonzero(rho2 > 1)[0]
    x3 = np.nonzero(rho3 > 1)[0]

    # Combine all detected indices and remove duplicates
    correl_indices = np.sort(np.unique(np.concatenate((x1, x2, x3))))

    return correl_indices
