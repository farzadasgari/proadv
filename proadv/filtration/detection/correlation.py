import numpy as np
from proadv.filtration.detection.poincare import calculate_ab, calculate_rho
from proadv.statistics.spread import std


def calculate_parameters(up, vp, wp):
    """
    Calculate parameters required for velocity correlation.

    Parameters
    ------
        up (array_like): Array of the first velocity component.
        vp (array_like): Array of the second velocity component.
        wp (array_like): Array of the third velocity component.

    Returns
    ------
        lambda_ (float): Lambda value used in velocity correlation detection.
        std_u (float): Standard deviation of the longitudinal velocity component.
        std_v (float): Standard deviation of the transverse velocity component.
        std_w (float): Standard deviation of the vertical velocity component.
    """
    data_size = up.size
    std_u = std(up)
    std_v = std(vp)
    std_w = std(wp)
    lambda_ = np.sqrt(2 * np.log(data_size))
    return lambda_, std_u, std_v, std_w


def velocity_correlation(ui, vi, wi):
    """
    Detect spikes using velocity correlation filter, based on three velocity components.

    Parameters
    ------
        ui (array_like): Array of the longitudinal velocity component.
        vi (array_like): Array of the transverse velocity component.
        wi (array_like): Array of the vertical velocity component.

    Returns
    ------
        correl_indices (array_like): Indices of spikes detected by velocity correlation.

    References
    ------
        Cea, L., J. Puertas, and L. Pena.
            "Velocity measurements on highly turbulent free surface flow using ADV."
            Experiments in fluids 42 (2007): 333-348.
    """
    from proadv.statistics.descriptive import mean
    ui, vi, wi = ui - mean(ui), vi - mean(vi), wi - mean(wi)
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
