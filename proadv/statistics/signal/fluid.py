from proadv.statistics.spread import mean

def kinetic_turbulent_energy(u, v, w):
    """
    Compute the kinetic turbulent energy based on velocity components.

    Parameters
    ------
    u (array_like): Array containing longitudinal velocity component. 
    v (array_like): Array containing transverse velocity component. 
    w (array_like): Array containing vertical velocity component. 

    Returns
    ------
    kinetic (float): Kinetic turbulent energy.
    """

    # Compute fluctuations
    up = u - mean(u)
    vp = v - mean(v)
    wp = w - mean(w)

    # Calculate the mean squared velocities in each direction
    mean_ui2 = mean(up ** 2)
    mean_vi2 = mean(vp ** 2)
    mean_wi2 = mean(wp ** 2)

    # Compute the total kinetic turbulent energy
    kinetic = 0.5 * (mean_ui2 + mean_vi2 + mean_wi2)

    return kinetic
