import numpy as np
from scipy import optimize


def _rotation(x, y):
    """
    Calculate the rotation angle theta between two variables x and y.

    Parameters
    ------
    x (array_like): Array containing the values of the first variable.
    y (array_like): Array containing the values of the second variable.

    Returns
    ------
    theta (float): The rotation angle theta in radians.
    """

    # Compute the size of the data
    data_size = x.size

    # Calculate the numerator and denominator for the rotation angle
    numerator = data_size * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = data_size * np.sum(x * x) - np.sum(x) * np.sum(x)

    # Compute the rotation angle theta using arctan2
    theta = np.arctan2(numerator, denominator)
    
    return theta


def _scaling(data):
    """
    Compute the scaling parameters for each dimension of the input data.

    Parameters
    ------
    data (array_like): Input data array.

    Returns
    ------
    max_co (array_like): Array containing the maximum value for each dimension.
    min_co (array_like): Array containing the minimum value for each dimension.
    scale (array_like): Array containing the scaling factor for each dimension.
    """

    # Compute the maximum and minimum values for each dimension
    max_co = data.max(1)
    min_co = data.min(1)

    # Compute the scaling factor for each dimension
    scale = max_co - min_co
    
    return max_co, min_co, scale


def _transform(data, max_co, min_co, scale):
    """
    Transform the input data using the provided scaling parameters.

    Parameters
    ------
    data (array_like): Input data array.
    max_co (array_like): Array containing the maximum value for each dimension.
    min_co (array_like): Array containing the minimum value for each dimension.
    scale (array_like): Array containing the scaling factor for each dimension.

    Returns
    ------
    transformed_data (array_like): Transformed data array.
    """
    
    # Compute the numerator and denominator for transformation
    numerator = data.T - np.tile(min_co, (data[0].size, 1))
    denominator = np.tile(scale, (data[0].size, 1))
    
    # Perform the transformation
    transformed_data = numerator / denominator
    
    return transformed_data


def _accumarray(subs, vals, sz):
    """
    Accumulate values into an array based on subscripts.

    Parameters
    ------
    subs (array_like): Subscripts indicating the position where each value should be accumulated.
    vals (array_like): Values to be accumulated.
    sz (tuple): Size of the output array.

    Returns
    ------
    accum (array_like): Accumulated array with the specified size.
    """
    
    # Initialize an array for accumulation with zeros
    accum = np.zeros(sz, dtype=vals.dtype)
    
    # Iterate over each subscript and accumulate the corresponding value
    for i, sub in enumerate(subs):
        accum[tuple(sub)] += vals[i]
        
    return accum


def _histogram(trans, grid):
    """
    Compute the histogram of the transformed data.

    Parameters
    ------
    trans (array_like): Transformed data array.
    grid (int): Number of bins along each dimension.

    Returns
    ------
    binned_data (array_like): Histogram of the transformed data.
    """
    
    # Get the number of rows and columns in the transformed data
    rows, cols = trans.shape
    
    # Initialize an array to store the bins
    bins = np.zeros((rows, cols), dtype=int)
    
    # Generate the histogram bins
    hist = np.linspace(0, 1, grid + 1)
    
    # Iterate over each column and compute the bins
    for i in range(cols):
        bins[:, i] = np.digitize(trans[:, i], hist, 1)
        bins[:, i] = np.minimum(bins[:, i], grid - 1)
        
    # Accumulate the binned data
    binned_data = _accumarray(bins, np.ones(rows), (grid,) * cols) / rows
    
    return binned_data


def _discrete_cosine_1d(data, weight):
    """
    Compute the 1D discrete cosine transform of the input data.

    Parameters
    ------
    data (array_like): Input data array.
    weight (float): Weight factor for the transform.

    Returns
    ------
    transform (array_like): 1D discrete cosine transform of the input data.
    """
    
    # Reorder the data for the discrete cosine transform
    reordered = np.vstack((data[::2, :], data[::-2, :]))
    
    # Compute the discrete cosine transform using FFT
    transform = np.real(weight * np.fft.fft(reordered))
    
    return transform


def _discrete_cosine_2d(data):
    """
    Compute the 2D discrete cosine transform of the input square data.

    Parameters
    ------
    data (array_like): Input square data array.

    Returns
    ------
    discrete (array_like): 2D discrete cosine transform of the input data.
    """
    
    # Get the number of rows and columns in the data
    rows, columns = data.shape
    
    # Check if the data shape is square
    if rows != columns:
        raise ValueError('Data shape must be square')
    
    # Generate the weight factors for the transform
    indices = np.arange(1, rows)
    w = np.concatenate(([1], 2 * np.exp(-1j * indices * np.pi / (2 * rows))))
    weight = np.tile(w[:, np.newaxis], (1, columns))
    
    # Compute the 1D discrete cosine transform for each row and column
    discrete = _discrete_cosine_1d(_discrete_cosine_1d(data, weight).T, weight).T
    
    return discrete


def _k(s_indices):
    """
    Compute the k-value based on the given indices.

    Parameters
    ------
    s_indices (int): The value of s indices.

    Returns
    -------
    k (float): The computed k-value.
    """
    
    # Define the step for the index array
    step = 2
    
    # Generate the index array
    index_array = np.arange(start=1, stop=2 * s_indices - 1 + 0.1 * step, step=step)
    
    # Compute and return the k-value
    return (-1) ** s_indices * np.prod(index_array) / np.sqrt(2 * np.pi)


def _psi(s_indices, time, initial_condition, autocorrelation_squared):
    weight_vector = np.exp(-initial_condition * np.pi ** 2 * time) * np.append(1, 0.5 * np.ones(
        len(initial_condition) - 1))
    wx = weight_vector * (initial_condition ** s_indices[0])
    wy = weight_vector * (initial_condition ** s_indices[1])
    result = (
            (-1) ** np.sum(s_indices)
            * (np.matmul(np.matmul(wy, autocorrelation_squared), wx.T))
            * np.pi ** (2 * np.sum(s_indices))
    )
    return result


def _evolve(t_guess, data_size: int, initial_condition, autocorrelation_squared):
    def __func(s, t):
        return _func(s, t, data_size, initial_condition, autocorrelation_squared)

    sum_func = __func([0, 2], t_guess) + __func([2, 0], t_guess) + 2 * __func([1, 1], t_guess)
    actual_time = (2 * np.pi * data_size * sum_func) ** (-1 / 3)
    time_evolution = (t_guess - actual_time) / actual_time
    return time_evolution, actual_time


def _func(s, t, n_sample: int, initial_condition, autocorrelation_squared):
    if sum(s) <= 4:
        sum_func = _func([s[0] + 1, s[1]], t, n_sample=n_sample, initial_condition=initial_condition,
                         autocorrelation_squared=autocorrelation_squared) + _func(
            [s[0], s[1] + 1], t, n_sample=n_sample, initial_condition=initial_condition,
            autocorrelation_squared=autocorrelation_squared
        )
        const = (1 + 1 / 2 ** (np.sum(s) + 1)) / 3
        time = (-2 * const * _k(s[0]) * _k(s[1]) / n_sample / sum_func) ** (1 / (2 + np.sum(s)))
        out = _psi(s, time, initial_condition, autocorrelation_squared)
    else:
        out = _psi(s, t, initial_condition, autocorrelation_squared)
    return out


def root(fun, n):
    max_tol = 0.1
    n = 50 * int(n <= 50) + 1050 * int(n >= 1050) + n * int((n < 1050) & (n > 50))
    tol = 10 ** -12 + 0.01 * (n - 50) / 1000
    solved = False
    while not solved:
        try:
            t = optimize.brentq(f=fun, a=0, b=tol)
            solved = True
        except ValueError:
            tol = min(tol * 2, max_tol)
        if tol >= max_tol:
            t = optimize.fminbound(func=lambda x: abs(fun(x)), x1=0, x2=0.1)
            solved = True
    return t


def bivariate_kernel(data, hx, hy, grid):
    data_size = data.size
    max_co, min_co, scale = _scaling(data)
    transformed_data = _transform(data, max_co, min_co, scale)
    binned_data = _histogram(transformed_data, grid)
    discrete = _discrete_cosine_2d(binned_data)
    ic = np.arange(0, discrete.shape[0], 1, dtype=float) ** 2
    ac2 = discrete ** 2
    t_star = root(lambda t: t - _evolve(t, data_size, ic, ac2)[0], n=data_size)

    def _temp(s, t):
        return _func(s, t, data.shape[1], ic, ac2)

    p_02 = _temp([0, 2], t_star)
    p_20 = _temp([2, 0], t_star)
    p_11 = _temp([1, 1], t_star)
    # t_y = (p_02 ** (3 / 4) / (4 * np.pi * data.shape[1] * p_20 ** (3 / 4) * (p_11 + np.sqrt(p_20 * p_02)))) ** (1 / 3)
    # t_x = (p_20 ** (3 / 4) / (4 * np.pi * data.shape[1] * p_02 ** (3 / 4) * (p_11 + np.sqrt(p_20 * p_02)))) ** (1 / 3)
    t_y = hy ** 2
    t_x = hx ** 2
    n_range = np.arange(0, grid, dtype=float)
    v1 = np.atleast_2d(np.exp(-(n_range ** 2) * np.pi ** 2 * t_x / 2)).T
    v2 = np.atleast_2d(np.exp(-(n_range ** 2) * np.pi ** 2 * t_y / 2))
    a_t = np.matmul(v1, v2) * discrete
    density_mx = _discrete_cosine_2d(a_t) * (a_t.size / np.prod(scale))
    density_mx[density_mx < 0] = np.finfo(float).eps
    x_step = scale[0] / (grid - 1)
    y_step = scale[1] / (grid - 1)
    x_vec = np.arange(start=min_co[0], stop=max_co[0] + 0.1 * x_step, step=x_step)
    y_vec = np.arange(start=min_co[1], stop=max_co[1] + 0.1 * y_step, step=y_step)
    x_mx, y_mx = np.meshgrid(x_vec, y_vec)
    density_mx = density_mx.T
    return density_mx, x_mx, y_mx
