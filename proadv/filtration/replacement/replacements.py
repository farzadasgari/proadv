import numpy as np
from proadv.statistics.descriptive import mean, median
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def last_valid_data(velocities, spike_indices):
    """
    Replace spike values in velocities array with the last valid data before each.

    Parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by last valid values.
        An array containing the modified data.
    """
    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Replace values at spikes indices with the last valid value
    for idx in spike_indices:
        modified_data[idx] = modified_data[idx - 1]

    return modified_data


def mean_value(velocities, spike_indices):
    """
    Replace spike values in velocities array with the mean value of velocity component.

    Parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by mean value of velocity component.
        An array containing the modified data.
    """

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Replace values at spikes indices with the mean value
    modified_data[spike_indices] = mean(velocities)

    return modified_data


def median_value(velocities, spike_indices):
    """
    Replace spike values in velocities array with the median value of velocity component.

    Parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by median value of velocity component.
        An array containing the modified data.
    """

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Replace values at spikes indices with the median value
    modified_data[spike_indices] = median(velocities)

    return modified_data


def linear_interpolation(velocities, spike_indices, decimals=4):
    """
    Perform linear interpolation to replace spike values in velocity data.

    This function identifies spikes in the velocity data and replaces them with
        linearly interpolated values based on the nearest valid data points before
        and after each spike. If a spike occurs at the end of the data, it is replaced
        with the mean of the entire dataset. The function is robust to handle individual
        spikes as well as sequences of consecutive spikes.

    Parameters
    ------
    velocities (array_like):  An array-like object containing velocity values. It should be a one-dimensional
        array of numerical data representing velocities.
    spike_indices (array_like): An array-like object containing the indices of detected spike events. It should
        be a one-dimensional array of integers where each integer represents the index
        in 'velocities' that corresponds to a spike.
    decimals (int, optional): The number of decimal places to round the interpolated values to. This allows
        the output data to be presented with a consistent level of precision. The default
        value is 4, but this can be adjusted as needed.

    Returns
    ------
    modified_data (array_like): An array containing the modified velocity data with spikes replaced by
        interpolated values. The shape and type of the array are the same as the
        input 'velocities' array.

    Notes
    ------
    The function uses linear interpolation to estimate the values of spikes based on the
        surrounding non-spike data. For spikes at the beginning or end of the data where a
        neighboring non-spike value is not available, the function uses the mean of the entire
        dataset as a replacement value. This approach ensures that the data remains as accurate
        and representative of the original dataset as possible.

    """

    # Create a copy of the velocity data to avoid modifying the original array
    modified_data = np.copy(velocities)

    # Initialize spike values with NaN to facilitate interpolation
    modified_data[spike_indices] = np.nan

    # Calculate differences between consecutive spike indices
    spike_diff = np.diff(spike_indices)
    # Identify the start index of each spike sequence
    spike_starts = spike_indices[np.insert(spike_diff > 1, 0, True)]
    # Identify the end index of each spike sequence
    spike_ends = spike_indices[np.append(spike_diff > 1, True)]
    # Iterate over each spike sequence for interpolation
    for start, end in zip(spike_starts, spike_ends):
        # Find the valid data point immediately before the spike sequence
        valid_start = modified_data[start - 1] if start > 0 else np.nan
        # Find the valid data point immediately after the spike sequence
        valid_end = modified_data[end + 1] if end < len(velocities) - 1 else np.nan

        # If valid start and end points are found, perform linear interpolation
        if not np.isnan(valid_start) and not np.isnan(valid_end):
            # Calculate the interpolation step size
            step = (valid_end - valid_start) / (end - start + 2)
            # Apply linear interpolation across the spike sequence
            for i, idx in enumerate(range(start, end + 1)):
                modified_data[idx] = valid_start + step * (i + 1)
        elif np.isnan(valid_end):  # Handle spike sequences at the end of the data
            # Replace end spikes with the mean of non-NaN values in the dataset
            modified_data[start: end + 1] = mean(velocities[~np.isnan(velocities)])
        else:
            # Use the mean of valid start and end points as a fallback
            fallback_value = np.nanmean([valid_start, valid_end])
            modified_data[start: end + 1] = fallback_value

    # Round the interpolated values to the specified number of decimal places
    return np.around(modified_data, decimals=decimals)


def cubic_12points_polynomial(velocities, spike_indices, decimals=4):
    """
    Interpolates missing data points in a velocity array using cubic polynomial interpolation.

    Parameters
    ------
    velocities (array_like): Array of velocity data. It should be a one-dimensional array-like object.
        This function assumes the input velocities array has at least 25 data points.
    spike_indices (array_like): Indices where data is missing (spikes). It should be a one-dimensional array-like
        object containing integers representing the indices of missing data points (spikes).
        If spike_indices contains invalid indices (e.g., negative values or indices exceeding the array size),
        a ValueError will be raised.
    decimals (int, optional): Number of decimal places to round the result to. Default is 4.

    Returns
    ------
    modified_data (array_like): Array of velocities with missing data interpolated. The interpolated values
        are calculated based on cubic polynomial interpolation using neighboring data points.

    Raises
    ------
    ValueError: If spike_indices contains invalid indices or if velocities is not a one-dimensional array-like object.

    Notes
    ------
    - This function uses cubic polynomial interpolation to estimate missing data points based on neighboring values.
    - If a missing data point (spike) occurs near the boundaries of the velocities array, linear interpolation
        is used instead of cubic polynomial interpolation.
    - The input velocities array is modified in-place to replace missing data points with interpolated values.
    """

    # Make a copy of velocities to preserve original data
    modified_data = velocities.copy()

    # Replace spike indices with NaN values
    modified_data[spike_indices] = np.nan

    # Generate x values for interpolation
    x = np.array(list(range(1, 13)) + list(range(14, 26)))

    for i in spike_indices:
        # Check if index is near the boundaries
        if i <= 30 or i >= (len(velocities) - 30):
            # Use linear interpolation near the boundaries
            modified_data[i] = np.around((velocities[i - 1] + modified_data[i:][~np.isnan(modified_data[i:])][0]) / 2,
                                         4)
        else:
            # Use cubic polynomial interpolation
            yint = np.delete(np.append(velocities[i - 13: i], modified_data[i:][~np.isnan(modified_data[i:])][0:12]),
                             12)
            f = interp1d(x, yint, 3)
            modified_data[i] = f(13)
    return np.around(modified_data, decimals=decimals)


def simple_moving_average(velocities, spike_indices, window_size=20):
    """
    Parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.
    window_size (int, optional): The size of the window for the moving average.
       Defaults to 20. Must be less than or equal to the size of the data array.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by simple_movingaverage of velocity component.
       An array containing the modified data.
    """

    from proadv.statistics.series import moving_average as ma

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Use moving_average function
    sma = ma(modified_data, window_size)

    # Replace values at spikes indices with the simple_moving_average values
    modified_data[spike_indices] = sma[spike_indices]
    return modified_data


def exponential_moving_average(velocities, spike_indices, alpha=0.2):
    """
    Parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.
    alpha (float, optional): Smoothing factor between 0 and 1.
        Higher alpha discounts older observations faster. Default is 0.2.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by exponential_moving_average of velocity component.
       An array containing the modified data.
    """

    from proadv.statistics.series import exponential_moving_average as expo

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Use exponential_moving_average function
    ema = expo(modified_data, alpha)

    # Replace values at spikes indices with the exponential_moving_average values
    modified_data[spike_indices] = ema[spike_indices]
    return modified_data


def weighted_moving_average(velocities, spike_indices, period=20):
    """
    Parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.
    period (int, optional): The period for the weighted moving average. Defaults to 20.
       Must be less than or equal to the size of the data array.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by weighted_moving_average of velocity component.
       An array containing the modified data.
    """

    from proadv.statistics.series import weighted_moving_average as wm

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Use weighted_moving_average function
    wma = wm(modified_data, period)

    # Replace values at spikes indices with the weighted_moving_average values
    modified_data[spike_indices] = wma[spike_indices]
    return modified_data


def ssa(velocities, spike_indices, fs, f):
    """
    parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.
    fs (float/int): Sampling frequency of the signal.
    f (float/int): maximum frequency of the signal of interest.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by singular spectrum analysis of velocity component.
       An array containing the modified data.
    """

    from proadv.statistics.series import ssa as sa

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Use ssa function
    xf = sa(modified_data, fs, f)

    # Replace values at spikes indices with the singular spectrum analysis values
    modified_data[spike_indices] = xf[spike_indices]
    return modified_data


def kalman_filter(velocities, spike_indices, initial_state, initial_covariance, process_noise, measurement_noise):
    """
    parameters
    ------
    velocities (array_like): Array of velocity data.
        An array-like object containing velocity values.
    spike_indices (array_like): Indices of spikes.
        An array-like object containing the indices of detected spikes.
    initial_state (array_like): An initial estimate for the state variable.
    initial_covariance (array_like): An initial estimate for the covariance.
    process_noise (array_like): Process noise that occurs in the process of changing a state variable.
    measurement_noise (array_like): Measurement noise present in the input data.

    Returns
    ------
    modified_data (array_like): Modified data with spikes replaced by kalman_filter of velocity component.
       An array containing the modified data.
    """

    from proadv.statistics.series import kalman_filter as kl

    # Create a copy of the original data
    modified_data = np.copy(velocities)

    # Use kalman_filter function
    filtered_data = kl(modified_data, initial_state, initial_covariance, process_noise, measurement_noise)

    # Replace values at spikes indices with the kalman_filter values
    modified_data[spike_indices] = filtered_data[spike_indices]


def _create_model(velocities, velocities_indices, spike_indices, degree):
    """
        This function is used to build a prediction model and find the next possible fit value

        Parameters
        ------
        velocities(array_like): The values we want to use to build the model.
        velocities_indices(array_like): The limit that we want our model to be made from.
        spike_indices(int): The amount we want to predict.
        degree(int): It specifies that our function should be polynomial

        Returns
        ------
        y_pred(float): The value is predicted

    """

    poly = PolynomialFeatures(degree=degree)
    poly_data = poly.fit_transform(velocities_indices.reshape(-1, 1))
    model = LinearRegression()
    model.fit(poly_data, velocities)
    x_poly = poly.fit_transform(spike_indices.reshape(-1, 1))
    y_pred = model.predict(x_poly)
    return y_pred


def polynomial_replacement(velocities, spike_indices, window=100, degree=2, decimals=4):
    """
        This function is used to predict appropriate values and replace them with inappropriate values.
        Using this function, you can use simple linear regression and functions of degree nth to replace
            and find the next optimal value.
        Note:
            - The optimal value for window and degree is 100 and 2, respectively.

        Parameters
        ------
        velocities(array_like): The original data has inappropriate values.
        spike_indices(array_like): Inappropriate data index in the main data.
        window(int,optional): The size we want to find the right model in that range.
        degree(int,optional): Specifies the degree of the function.
        decimals(float,optional): Specifies the number of digits to round.

        Returns
        ------
        modified_data(array_like): The final data after running the algorithm and replacement.
    """

    # Make a copy of velocities to preserve original data
    modified_data = velocities.copy()
    a = 0
    for i in spike_indices.squeeze():
        if i == 0:
            # If the first index of the data set is a spike, it will be replaced with the value of the mean of the data.
            modified_data[i] = np.mean(modified_data)
        elif 1 <= i <= 10:
            # Spikes 1 to 10 are replaced using linear interpolation algorithm
            if a == 0:
                modified_data = linear_interpolation(modified_data,
                                                     spike_indices[np.where(spike_indices < 10)[0]].squeeze(), decimals)
                a += 1  # "a" is to run this algorithm only once.
        elif 11 <= i <= window - 1:
            # Predicting the appropriate spike value for indexes less than window
            modified_data[i] = _create_model(modified_data[:i], np.arange(i), i, degree=degree).squeeze()
        else:
            # Predicting the appropriate spike value for indexes whose size is larger than the window
            modified_data[i] = _create_model(modified_data[i - window: i], np.arange(i - window, i), i,
                                             degree).squeeze()
    modified_data = np.around(modified_data, decimals=decimals)
    return modified_data
