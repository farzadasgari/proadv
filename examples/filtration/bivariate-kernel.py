from proadv.filtration.detection.bivariatekernel import _cutoff, _derivative
from proadv.kernel.bivariate import _rotation
from proadv.filtration.replacement.replacements import linear_interpolation
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde


def main():
    """
    Example function demonstrating the use of bivariate kernel density detection for data filtration.

    Reads a CSV file containing velocity data, applies bivariate kernel density detection iteratively,
    and plots the filtered results.
    """

    # Read velocity data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values

    # Calculate derivative of the velocity data
    derivative = _derivative(main_data)

    # Determine rotation angle for transformation
    theta = _rotation(main_data, derivative)

    # Transform velocity data
    ut = main_data * np.cos(theta) + derivative * np.sin(theta)
    wt = -main_data * np.sin(theta) + derivative * np.cos(theta)

    # Generate grid for kernel density estimation
    grid = 256
    xmin = ut.min()
    xmax = ut.max()
    ymin = wt.min()
    ymax = wt.max()
    X, Y = np.mgrid[xmin:xmax:grid * 1j, ymin:ymax:grid * 1j]
    uf = X[:, 0]
    wf = Y[0, :]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([ut, wt])

    # Perform kernel density estimation
    kernel = kde(values)
    density = np.reshape(kernel(positions).T, X.shape)
    dp = density.max()
    up, wp = np.where(density == dp)[0][0], np.where(density == dp)[1][0]
    fu = density[:, wp]
    fw = density[up, :]

    # Determine cutoff values
    ul, uu = _cutoff(dp, uf, 0.4, 0.4, fu, up, grid)
    wl, wu = _cutoff(dp, wf, 0.4, 0.4, fw, wp, grid)
    uu1 = uu - 0.5 * (uu + ul)
    ul1 = ul - 0.5 * (uu + ul)
    wu1 = wu - 0.5 * (wu + wl)
    wl1 = wl - 0.5 * (wu + wl)
    Ut1 = ut - 0.5 * (uu + ul)
    Wt1 = wt - 0.5 * (wu + wl)

    # Initialize array for spike detection
    F = np.zeros(main_data.size)
    at = 0.5 * (uu1 - ul1)
    bt = 0.5 * (wu1 - wl1)

    # Perform spike detection
    for i in range(0, main_data.size):
        if Ut1[i] > uu1 or Ut1[i] < ul1:
            F[i] = 1
        else:
            we = np.sqrt((bt ** 2) * (1 - (Ut1[i] ** 2) / (at ** 2)))
            if Wt1[i] > we or Wt1[i] < -we:
                F[i] = 1

    # Identify spike indices
    indices = np.where(F > 0)[0]

    # Perform data interpolation for spike removal
    filtered_data = linear_interpolation(main_data, indices)

    # Plotting the filtered and unfiltered data
    sampling_frequency = 100  # Hz
    delta_time = 1 / sampling_frequency
    time = np.arange(0, main_data.size * delta_time, delta_time)
    plt.plot(time, main_data, color='crimson', label='Unfiltered')
    plt.plot(time, filtered_data, color='black', label='Filtered')
    plt.title('Bivariate Kernel Density')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
