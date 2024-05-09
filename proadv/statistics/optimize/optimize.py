import numpy as np


def brentq(f, xa, xb, xtol=1.48e-8, rtol=1.48e-8, maxiter=500, full_output=False):
    """
    Find a root of a continuous function within a given interval using Brent's method.

    Brent's method combines bisection, secant, and inverse quadratic interpolation. It has
    the reliability of bisection but it can be as quick as some of the less reliable methods.

    Parameters
    ------
    f : callable
        The function whose root is to be estimated. The function must be continuous and
        f(xa) and f(xb) must have opposite signs.
    xa, xb : float
        The interval within which to search for a root. The function f(xa) and f(xb) must
        have opposite signs.
    xtol : float, optional
        The convergence tolerance for the root. Defaults to 1.48e-8.
    rtol : float, optional
        The relative tolerance for the root. Defaults to 1.48e-8.
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 500.
    full_output : bool, optional
        If True, return a dictionary containing the root, number of function calls,
        number of iterations, a boolean indicating if the algorithm converged, and
        a flag describing the convergence status. If False, return only the root.

    Returns
    ------
    root : float
        Estimated root location.
    result : dict, optional
        A dictionary containing:
        - 'root': the root of the function,
        - 'function_calls': number of function calls made,
        - 'iterations': number of iterations performed,
        - 'converged': boolean flag indicating if the algorithm converged,
        - 'flag': description of the convergence status.
        This output is returned only if `full_output` is True.

    Raises
    ------
    ValueError
        If `xa` is not less than `xb`, or if `f(xa)` and `f(xb)` do not have opposite signs,
        or if the algorithm fails to converge within `maxiter` iterations.

    Examples
    ------
    >>> def example_function(x):
    ...     return x**2 - 2
    >>> root = brentq(example_function, 0, 2)
    >>> print(root)
    1.4142135623730951
    """
    # Check if the initial interval is valid
    if xa >= xb:
        raise ValueError("xa must be less than xb for the bracketing interval.")

    # Evaluate the function at the endpoints of the interval
    xpre = xa
    xcur = xb
    fpre = f(xpre)
    fcur = f(xcur)

    # Check if we already have a root at the endpoints
    if fpre == 0:
        return (xpre, 0, 0, 1) if full_output else xpre
    if fcur == 0:
        return (xb, 0, 0, 1) if full_output else xb

    # Ensure the function has opposite signs at the endpoints (necessary condition for Brent's method)
    if np.sign(fpre) == np.sign(fcur):
        raise ValueError("Function must have opposite signs at the endpoints xa and xb.")

    # Initialize variables for the Brent's method
    xblk, fblk, spre, scur, i = 0, 0, 0, 0, 0

    # Main iteration loop
    for i in range(maxiter):
        # Check if the current interval [xpre, xcur] brackets the root
        if np.sign(fpre) != np.sign(fcur):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        # Swap if necessary to ensure that fcur is closer to zero than fpre
        if abs(fblk) < abs(fcur):
            xpre, xcur = xcur, xblk
            xblk = xpre
            fpre, fcur = fcur, fblk
            fblk = fpre

        # Calculate the tolerance
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2

        # Check for convergence
        if fcur == 0 or abs(sbis) < delta:
            result = {'root': xcur, 'function_calls': i + 2, 'iterations': i + 1, 'converged': True,
                      'flag': 'converged'}
            return result if full_output else xcur

        # Determine whether to perform bisection or interpolation/extrapolation
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # Interpolation
                stry = -fcur * (xcur - xpre) / (fcur - fpre)

            else:
                # Extrapolation
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))

            # Decide if we accept the interpolation/extrapolation
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                # Accept the interpolation/extrapolation step
                spre = scur
                scur = stry
            else:
                # Reject and bisect instead
                spre = sbis
                scur = sbis
        else:
            # Not enough decrease, so bisect
            spre = sbis
            scur = sbis

        # Update the current point
        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur  # Increment the current point by the step size
        else:
            xcur += delta if sbis > 0 else -delta  # Ensure the step is big enough

        # Evaluate the function at the new current point
        fcur = f(xcur)

    # If we reach here, the method did not converge within the maximum number of iterations
    result = {
        'root': xcur,
        'function_calls': i + 2,
        'iterations': i + 1,
        'converged': False,
        'flag': 'convergence not achieved'
    }
    if full_output:
        return result
    else:
        return result['root']


def _optimize_within_bounds(target_function, limit_bounds, extra_params=(),
                            convergence_tol=1e-5, iteration_limit=500):
    """
    Minimize a scalar function within bounds using Brent's method.

    This function implements Brent's method for finding a local minimum of a scalar function
    within a given interval. It combines a golden-section search with parabolic interpolation
    for efficiency and reliability.

    Parameters
    ----------
    target_function : callable
        The objective function to minimize. Must be callable with the signature f(x, *args).
    limit_bounds : tuple
        A tuple (lower_bound, upper_bound) specifying the interval within which to search.
    extra_params : tuple, optional
        Additional arguments to pass to the target_function.
    convergence_tol : float, optional
        The convergence tolerance. Optimization stops when the change in the function value
        between iterations is less than or equal to this value.
    iteration_limit : int, optional
        The maximum number of iterations allowed before stopping the algorithm.

    Returns
    -------
    x_optimal : float
        The point at which the function is minimized within the bounds.
    fval : float
        The function value at x_optimal.
    flag : int
        An integer flag indicating the status of the optimization (0 if successful,
        1 if the iteration limit was reached, 2 if NaN was encountered).
    iteration_count : int
        The number of iterations performed.
    """

    # Set the maximum number of function evaluations to the iteration limit
    max_function_evals = iteration_limit

    # Check that the bounds are a tuple with two elements
    if len(limit_bounds) != 2:
        raise ValueError('Bounds should consist of two elements.')

    # Unpack the lower and upper bounds from the tuple
    lower_limit, upper_limit = limit_bounds

    # Ensure both bounds are finite numbers
    if not (np.isfinite(lower_limit) and np.isfinite(upper_limit)):
        raise ValueError("Bounds must be finite numbers.")

    # Verify that the lower bound is less than the upper bound
    if lower_limit > upper_limit:
        raise ValueError("Lower bound must be less than upper bound.")

    # Initialize the status flag to 0, indicating no errors
    convergence_flag = 0

    # Calculate the square root of the machine epsilon for floating-point arithmetic
    epsilon_sqrt = np.sqrt(2.2e-16)

    # Define the golden ratio constant for the golden-section search
    golden_ratio = 0.5 * (3.0 - np.sqrt(5.0))

    # Initialize the bracketing interval [left, right] for the search
    left, right = lower_limit, upper_limit

    # Calculate the first point to evaluate by applying the golden ratio to the interval
    x_middle = left + golden_ratio * (right - left)

    # Initialize the points x_optimal and x_temp to x_middle
    x_optimal, x_temp = x_middle, x_middle

    # Initialize the step size and the variable 'e' used for parabolic steps
    step_size = e = 0.0

    # Evaluate the target function at the initial point
    x_current = x_optimal
    f_current = target_function(x_current, *extra_params)

    # Start the iteration count at 1
    iteration_count = 1

    # Initialize f_temp to infinity for comparison later
    f_temp = np.inf

    # Initialize f_x_middle and f_x_temp to the current function value
    f_x_middle = f_x_temp = f_current

    # Calculate the central point of the interval
    x_central = 0.5 * (left + right)

    # Calculate the tolerances used for convergence checks
    tolerance_1 = epsilon_sqrt * np.abs(x_optimal) + convergence_tol / 3.0
    tolerance_2 = 2.0 * tolerance_1

    # Start the main optimization loop
    while np.abs(x_optimal - x_central) > (tolerance_2 - 0.5 * (right - left)):
        is_golden = True  # Assume a golden-section step unless a parabolic step is taken

        # Attempt a parabolic fit if the previous step was large enough
        if np.abs(e) > tolerance_1:
            is_golden = False  # A parabolic step is attempted, so set is_golden to False
            # Calculate the coefficients for the parabolic fit
            r = (x_optimal - x_temp) * (f_current - f_x_middle)
            q = (x_optimal - x_middle) * (f_current - f_x_temp)
            p = (x_optimal - x_middle) * q - (x_optimal - x_temp) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            previous_step = e
            e = step_size

            # Check if the parabolic step is acceptable
            if np.abs(p) < np.abs(0.5 * q * previous_step) and q * (left - x_optimal) < p < q * (right - x_optimal):
                step_size = p / q
                x_current = x_optimal + step_size

                # Adjust the step size if x_current is too close to the interval boundaries
                if ((x_current - left) < tolerance_2) or ((right - x_current) < tolerance_2):
                    delta = np.sign(x_central - x_optimal) + (x_central == x_optimal)
                    step_size = tolerance_1 * delta
            else:
                is_golden = True  # The parabolic step is not acceptable, revert to golden-section

        # If a golden-section step is needed
        if is_golden:
            # Update 'e' for the golden-section step
            if x_optimal >= x_central:
                e = left - x_optimal
            else:
                e = right - x_optimal
            step_size = golden_ratio * e

        # Calculate the next point to evaluate by adding the step size to x_optimal
        delta = np.sign(step_size) + (step_size == 0)
        x_current = x_optimal + delta * np.maximum(np.abs(step_size), tolerance_1)
        f_temp = target_function(x_current, *extra_params)
        iteration_count += 1

        # Update the interval based on the function evaluation
        if f_temp <= f_current:
            if x_current >= x_optimal:
                left = x_optimal
            else:
                right = x_optimal
            x_middle, f_x_middle = x_temp, f_x_temp
            x_temp, f_x_temp = x_optimal, f_current
            x_optimal, f_current = x_current, f_temp
        else:
            if x_current < x_optimal:
                left = x_current
            else:
                right = x_current
            if (f_temp <= f_x_temp) or (x_temp == x_optimal):
                x_middle, f_x_middle = x_temp, f_x_temp
                x_temp, f_x_temp = x_current, f_temp
            elif (f_temp <= f_x_middle) or (x_middle == x_optimal) or (x_middle == x_temp):
                x_middle, f_x_middle = x_current, f_temp

        # Recalculate the central point and tolerances
        x_central = 0.5 * (left + right)
        tolerance_1 = epsilon_sqrt * np.abs(x_optimal) + convergence_tol / 3.0
        tolerance_2 = 2.0 * tolerance_1

        # Check if the iteration limit has been reached
        if iteration_count >= max_function_evals:
            convergence_flag = 1
            break

    # Check for NaN values in the function evaluations
    if np.isnan(x_optimal) or np.isnan(f_current) or np.isnan(f_temp):
        convergence_flag = 2

    # Set the final optimal value and function value
    x_optimal = x_current
    fval = f_current
    flag = convergence_flag

    # Return the results as a tuple
    return x_optimal, fval, flag, iteration_count


def fminbound(target, lower, upper, parameters=(), tolerance=1e-5, max_evaluations=500,
              detailed_output=False):
    """
    Minimize a scalar function within a bounded interval.

    Given a function of one variable and a bounded interval, `fminbound` finds a local
    minimum using a modification of Brent's method. The algorithm combines a bracketing
    strategy with a parabolic interpolation to efficiently find the minimum. It is robust
    and has the advantage of using a small number of function evaluations.

    Parameters
    ----------
    target : callable
        The objective function to minimize. Must be in the form f(x, *args).
    lower : float
        The lower bound of the interval for the minimization.
    upper : float
        The upper bound of the interval for the minimization.
    parameters : tuple, optional
        Additional arguments to pass to the `target` function. Default is an empty tuple.
    tolerance : float, optional
        The convergence tolerance. Optimization stops when the difference in the
        function value between iterations is less than or equal to this value.
        Default is 1e-5.
    max_evaluations : int, optional
        The maximum number of function evaluations allowed. Default is 500.
    detailed_output : bool, optional
        If True, return additional output information. Default is False.

    Returns
    -------
    optimum : float
        The point within the specified bounds where the `target` function attains its
        minimum value.
    function_value : float, optional
        The value of the `target` function at the `optimum`. Returned only if
        `detailed_output` is True.
    error_flag : int, optional
        An integer flag indicating the status of the optimization: 0 if the function
        converged to a solution and 1 if the maximum number of function evaluations
        was exceeded. Returned only if `detailed_output` is True.
    evaluations : int, optional
        The number of function evaluations made. Returned only if `detailed_output`
        is True.

    Examples
    --------
    >>> def quadratic(x):
    ...     return (x - 2)**2 + 1
    >>> fminbound(quadratic, -1, 5)
    2.0
    """

    # Call the internal bounded minimization function
    x_optimal, fval, flag, iteration_count = _optimize_within_bounds(target, (lower, upper), parameters,
                                                                     convergence_tol=tolerance,
                                                                     iteration_limit=max_evaluations)

    # Return the results based on the detailed_output flag
    if detailed_output:
        return x_optimal, fval, flag, iteration_count
    else:
        return x_optimal
