def _relative_scale(z_array, p_array):
    """
    Return the relative scale of the transfer function from zero and pole
    """
    scale = np.copy(p_array).shape[0] - np.copy(z_array).shape[0]
    if scale < 0:
        raise ValueError("Improper transfer function. Must have at least as many poles as zeros.")
    else:
        return scale


def _catenate(array, fpc):
    # To join multiple presentations

    catenate_array = np.concatenate((array + sqrt(array ** 2 - fpc ** 2),
                                     array - sqrt(array ** 2 - fpc ** 2)))
    return catenate_array


def low_to_stop(zero, poles, system, pc=1.0, pw=1.0):
    """
    This function converts a LP(low-pass) filter prototype to a BS(band-stop) filter.

    Parameters
    ------
    zero (array_like) : Zeros of the analog filter transfer function.
    poles (array_like) : Poles of the analog filter transfer function.
    system (float) : System gain of the analog filter transfer function.
    pc (float) : Desired stopband center, as angular frequency.
    pw (float) : Desired stopband width, as angular frequency.

    Returns
    ------
    zerob (array_like): Zeros of the transformed BS(band-stop) filter transfer function.
    polesb (array_like): Poles of the transformed BS(band-stop) filter transfer function.
    systemb (float) : System gain of the transformed BS(band-stop) filter.

    Examples
    ------
    >>> zero = [6 + 3j, 6 - 3j]
    >>> poles = [8, -20]
    >>> system = 0.5
    >>> pc = 0.53
    >>> pw = 14
    >>> low_to_stop(zero, poles, system, pc, pw)
    (array([1.74568173-1.00283509j, 1.74568173+1.00283509j,
           0.12098494+0.06950176j, 0.12098494-0.06950176j]), array([ 1.57122195+0.j        , -0.35      -0.39799497j,
            0.17877805+0.j        , -0.35      +0.39799497j]), -0.140625)

    """
    zero = atleast_1d(zero)
    poles = atleast_1d(poles)
    fpc = float(pc)
    fpw = float(pw)

    scale = _relative_scale(zero, poles)

    lf = lambda a, b: (a / 2) / b
    zerol = lf(fpw, zero)
    polesl = lf(fpw, poles)

    zerol = zerol.astype(complex)
    polesl = polesl.astype(complex)

    zerob = _catenate(zerol, fpc)
    polesb = _catenate(polesl, fpc)

    zerob = append(zerob, full(scale, +1j * fpc))
    zerob = append(zerob, full(scale, -1j * fpc))

    systemb = system * real(prod(-zero) / prod(-poles))

    return zerob, polesb, systemb

def low_to_band(zero, poles, system, pc=1.0, pw=1.0):
    """
    This function converts a LP(low-pass) filter prototype to a BP(band-pass) filter.

    Parameters
    ------
    zero (array_like) : Zeros of the analog filter transfer function.
    poles (array_like) : Poles of the analog filter transfer function.
    system (float) : System gain of the analog filter transfer function.
    pc (float) : Desired passband center, as angular frequency.
    pw (float) : Desired passband width, as angular frequency.

    Returns

    zerob (array_like): Zeros of the transformed BP(band-pass) filter transfer function.
    polesb (array_like): Poles of the transformed BP(band-pass) filter transfer function.
    systemb (float) : System gain of the transformed BP(band-pass) filter.
    ------

    Examples
    ------
    >>> zero = [6 + 3j, 6 - 3j]
    >>> poles = [8, -20]
    >>> system = 0.5
    >>> pc = 0.53
    >>> pw = 14
    >>> low_to_band(zero, poles, system, pc, pw)
    (array([8.39973247e+01+4.20013377e+01j, 8.39973247e+01-4.20013377e+01j,
           2.67525513e-03-1.33771277e-03j, 2.67525513e-03+1.33771277e-03j]), array([ 1.11997492e+02+0.j, -1.00321788e-03+0.j,  2.50809188e-03+0.j,
           -2.79998997e+02+0.j]), 0.5)
    """
    zero = atleast_1d(zero)
    poles = atleast_1d(poles)
    fpc = float(pc)
    fpw = float(pw)

    scale = _relative_scale(zero, poles)
    # Scale poles and zeros to desired bandwidth
    lf = lambda a, b: a * b / 2
    zerol = lf(zero, fpw)
    polesl = lf(poles, fpw)

    # Square root needs to produce complex result, not NaN
    zerol = zerol.astype(complex)
    polesl = polesl.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +pc and -pc
    zerob = _catenate(zerol, fpc)
    polesb = _catenate(polesl, fpc)

    zerob = append(zerob, zeros(scale))

    # Cancel out gain change from frequency scaling
    systemb = system * fpw ** scale

    return zerob, polesb, systemb