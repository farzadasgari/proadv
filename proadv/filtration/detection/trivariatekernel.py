import numpy as np
from scipy import linalg
from scipy.stats._stats import gaussian_kernel_estimate as gke
from scipy.stats import gaussian_kde as kde
from proadv.statistics.spread import std


def _derivatives(data):
    """
    Calculate time-independent first and second order derivatives of the input data.

    Parameters
    ------
    data (array_like): Input data.

    Returns
    ------
    dc (array_like): First derivative of the input data.
    dc2 (array_like): Second derivative of the input data.
    """
    
    # Initialize arrays for first and second derivatives
    dc = np.zeros_like(data)
    dc2 = np.zeros_like(data)

    # Calculate first derivative
    for i in range(1, data.size - 1):
        dc[i] = (data[i + 1] - data[i - 1]) / 2

    # Calculate second derivative
    for i in range(1, data.size - 1):
        dc2[i] = (dc[i + 1] - dc[i - 1]) / 2

    return dc, dc2


def _rotation(u1, w1):
    """
    Calculate rotation angle theta between vectors u1 and w1.

    Parameters
    ------
    u1 (array_like): Vector u1.
    w1 (array_like): Vector w1.

    Returns
    ------
    theta (float): Rotation angle.
    """

    # Compute data size
    data_size = u1.size

    # Calculate theta using the arctan2 function
    theta = np.arctan2((data_size * np.sum(u1 * w1) - np.sum(u1) * np.sum(w1)),
                       (data_size * np.sum(u1 * u1) - np.sum(u1) * np.sum(u1)))
    
    return theta


def _transform(x, y, theta):
    """
    Transform coordinates (x, y) using rotation angle theta.

    Parameters
    ------
    x (array_like): x-coordinate.
    y (array_like): y-coordinate.
    theta (float): Rotation angle

    Returns
    ------
    xt, yt (array_like): Transformed coordinates.
    """

    # Apply rotation transformation
    xt = x * np.cos(theta) + y * np.sin(theta)
    yt = -x * np.sin(theta) + y * np.cos(theta)

    return xt, yt


def _scaling(x, y, grid):
    """
    Scale coordinates (x, y) into a grid.

    Parameters
    ------
    x (array_like): x-coordinate.
    y (array_like): y-coordinate.
    grid (int): Number of grid points.

    Returns
    ------
    Meshgrid of scaled coordinates.
    """

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    return np.mgrid[xmin:xmax:grid * 1j, ymin:ymax:grid * 1j]


def _profile(meshgrid_x, meshgrid_y):
    return meshgrid_x[:, 0], meshgrid_y[0, :]


def _position(meshgrid_x, meshgrid_y, trans_x, trans_y):
    positions = np.vstack([meshgrid_x.ravel(), meshgrid_y.ravel()])
    
    values = np.vstack([trans_x, trans_y])

    return positions, values


def _factor(rows, cols):
    return np.power(cols, -1. / (rows + 4))


def _weight(cols):
    return np.ones(cols) / cols


def _cov(data, aweights):
    return np.atleast_2d(np.cov(data, rowvar=1, bias=False, aweights=aweights))


def _cholesky(data):
    return linalg.cholesky(data)


def _determination(data):
    return 2 * np.sum(np.log(np.diag(data * np.sqrt(np.pi * 2))))


def _covariance(data, rows, cols):
    factor = _factor(rows, cols)
    weight = _weight(cols)
    net = np.power(np.sum(weight ** 2), -1)
    factor = _factor(rows, net)
    cov = _cov(data, weight)
    sky = _cholesky(cov)
    covariance = cov * factor ** 2
    cholesky = sky * factor
    cholesky.dtype = np.float64
    log = _determination(cholesky)
    compute = {
        "factor": factor,
        "weight": weight,
        "net": net,
        "covariance": covariance,
        "cholesky": cholesky,
        "log": log
    }
    return compute


def _type(cov, scatt):
    data_type = np.common_type(cov, scatt)
    data_size = np.dtype(data_type).itemsize
    if data_size == 4:
        data_size = 'float'
    elif data_size == 8:
        data_size = 'double'
    elif data_size in (12, 16):
        data_size = 'long double'
    return data_type, data_size


def _density(values):
    dataset = np.atleast_2d(np.asarray(values))
    if dataset.size < 1:
        raise ValueError("Dataset should have more than one element.")
    rows, cols = dataset.shape
    if rows > cols:
        raise ValueError("Number of dimensions exceeds the number of samples.")
    evals = _covariance(dataset, rows, cols)
    return evals


def _estimation(kde, x):
    return np.reshape(kde, x.shape)


def _extraction(dataset, parameters, poses):
    scatt = np.atleast_2d(np.asarray(poses))
    data_type, data_mode = _type(parameters["covariance"], scatt)
    return dataset.T, parameters["weight"][:, None], poses.T, parameters["cholesky"].T, data_type, data_mode


def _evolve(dataset, poses, computations):
    dataset = _extraction(dataset, computations, poses)
    dens = gke[dataset[5]](
        dataset[0],
        dataset[1],
        dataset[2],
        dataset[3],
        dataset[4])
    return dens[:, 0]


def _peak(pdf):
    peak = pdf.max()
    up, wp = np.where(pdf == peak)[0][0], np.where(pdf == peak)[1][0]
    fu = pdf[:, wp]
    fw = pdf[up, :]
    return peak, up, wp, fu, fw


def _cutoff(dp, uf, c1, c2, f, Ip, ngrid):
    lf = f.size
    dk = np.append([0], np.diff(f)) * ngrid / dp
    for i in list(range(1, Ip))[::-1]:
        if f[i] / f[Ip] <= c1 and abs(dk[i]) <= c2:
            i1 = i
            break
        else:
            i1 = 1

    for i in range(Ip + 1, lf - 1):
        if f[i] / f[Ip] <= c1 and abs(dk[i]) <= c2:
            i2 = i
            break
        else:
            i2 = lf - 1
    ul = uf[i1]
    uu = uf[i2]
    return ul, uu


def kernel(u1, w1, grid):
    dataset = np.array([u1, w1])
    theta = _rotation(u1, w1)
    ut, wt = _transform(u1, w1, theta)
    mesh_u, mesh_w = _scaling(ut, wt, grid)
    uf, wf = _profile(mesh_u, mesh_w)
    positions, values = _position(mesh_u, mesh_w, ut, wt)
    evals = _density(values)
    evole = _evolve(dataset, positions, evals)
    estimation = _estimation(evole.T, mesh_u)
    density = np.reshape(kde(values)(positions).T, mesh_u.shape)
    peak, up, wp, fu, fw = _peak(density)
    lambda_ = np.sqrt(2 * np.log(u1.size))
    stdu = std(u1)
    stdw = std(w1)
    cu = lambda_ * stdu / np.sqrt(u1.size)
    cw = lambda_ * stdw / np.sqrt(u1.size)
    ul, uu = _cutoff(peak, uf, cu, cu, fu, up, grid)
    wl, wu = _cutoff(peak, wf, cw, cw, fw, wp, grid)
    uu1 = uu - 0.5 * (uu + ul)
    wu1 = wu - 0.5 * (wu + wl)
    Ut1 = ut - 0.5 * (uu + ul)
    Wt1 = wt - 0.5 * (wu + wl)
    rho = (Ut1 / uu1) ** 2 + (Wt1 / wu1) ** 2
    id_ = np.where(rho > 1)[0]
    return id_, estimation


def three_dimensional_kernel(data, grid):
    dc, dc2 = _derivatives(data)
    x1, _ = kernel(data, dc, grid)
    x2, _ = kernel(data, dc2, grid)
    x3, _ = kernel(dc, dc2, grid)
    kde3d_indices = np.sort(np.unique(np.concatenate((x1, x2, x3))))
    return kde3d_indices
