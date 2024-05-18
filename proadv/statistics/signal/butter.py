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

def low_to_high(zero, poles, system, dc=1.0):
    """
    This function converts a low pass filter prototype to another frequency.

    Parameters
    ------
    zero (array_like) : Zeros of the analog filter transfer function.
    poles (array_like) : Poles of the analog filter transfer function.
    system (float) : System gain of the analog filter transfer function.
    dc (float) : Desired cutoff, as angular frequency.

    Returns
    ------
    zeroh (array_like): Zeros of the transformed LP(low-pass) filter transfer function.
    polesh (array_like): Poles of the transformed LP(low-pass) filter transfer function.
    systemh (float) : System gain of the transformed LP(low-pass) filter.

    """
    zero = atleast_1d(zero)
    poles = atleast_1d(poles)
    fdc = float(dc)

    scale = _relative_scale(zero, poles)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    zeroh = fdc / zero
    polesh = fdc / poles

    # If lowpass had zeros at infinity, inverting moves them to origin.
    zeroh = append(zeroh, zeros(scale))

    # Cancel out gain change caused by inversion
    systemh = system * real(prod(-zero) / prod(-poles))

    return zeroh, polesh, systemh

def bilinear(zero, poles, system, sr):
    """
    This function returns a digital IIR filter using a bilinear conversion from an analog one.

    Parameters
    ----------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float) : System gain of the analog filter transfer function.
    sr (float) : Sample rate, as ordinary frequency.

    Returns
    -------
    zerob (array_like): Zeros of the transformed digital filter transfer function.
    polesb (array_like): Poles of the transformed digital filter transfer function.
    systemb (float) : System gain of the transformed digital filter.

    """
    zero = atleast_1d(zero)
    poles = atleast_1d(poles)

    fsr = _valid_g(sr, an=False)

    scale = _relative_scale(zero, poles)

    fsr2 = 2.0 * fsr

    # Bilinear transform the poles and zeros
    lf = lambda x, y: (x + y) / (x - y)
    zerob = lf(fsr2, zero)
    polesb = lf(fsr2, poles)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    zerob = append(zerob, -ones(scale))

    # Compensate for gain change
    systemb = system * real(prod(fsr2 - zero) / prod(fsr2 - poles))

    return zerob, polesb, systemb

def low_to_low(zero, poles, system, dc=1.0):
    """
    This function converts a LP(low-pass) filter prototype to another frequency.

    Parameters
    ----------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float) : System gain of the analog filter transfer function.
    dc (float) : Desired cutoff, as angular frequency.

    Returns
    -------
    zerol (array_like): Zeros of the transformed LP(low-pass) filter transfer function.
    polesl (array_like): Poles of the transformed LP(low-pass) filter transfer function.
    systeml (float) : System gain of the transformed LP(low-pass) filter.

    Examples
    ------
    >>> zero = [8, 3]
    >>> poles = [6, 14]
    >>> system = 0.7
    >>> dc = 0.5
    >>> low_to_low(zero, poles, system, dc)
    (array([4. , 1.5]), array([3., 7.]), 0.7)

    """
    zero = atleast_1d(zero)
    poles = atleast_1d(poles)
    fdc = float(dc)  # Avoid int wraparound

    scale = _relative_scale(zero, poles)

    # Scale all points radially from origin to shift cutoff frequency
    zerol = fdc * zero
    polesl = fdc * poles

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    systeml = system * fdc ** scale

    return zerol, polesl, systeml