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

def buttap(number):
    """
    Return (x, z, g) for analog prototype of Nth-order Butterworth filter.
    """
    if np.abs(int(number)) != number:
        raise ValueError("Filter order must be a nonnegative integer")
    x = numpy.array([])
    y = numpy.arange(-number + 1, number, 2)
    # Middle value is 0 to ensure an exactly real pole
    z = -numpy.exp(1j * pi * y / (2 * number))
    g = 1
    return x, z, g

def xyz_to_sop(x, y, z, pg=None, *, ag=False):
    """
    This function returns the second-order parts of the zero, pole, and gain of a system.

    Parameters
    ----------
    x (array-like): Zeros of the transfer function.
    y (array-like): Poles of the transfer function.
    z (float): System gain.
    pg: The method to use to combine pairs of poles and zeros into sections.
        The values it can have:
        - None(default)
        - nr(nearest)
        - kd(keep_odd)
        - ml(minimal)
    ag(bool): If False, the system is discrete, otherwise it is analog.

    Returns
    -------
    sop(array-like): Array of second-order filter coefficients, with shape (n_parts, 6).

    """
    if pg is None:
        pg = 'ml' if ag else 'nr'

    vp = ['nr', 'kd', 'ml']
    if pg not in vp:
        raise ValueError(f'pairing must be one of {vp}, not {pg}')

    if ag and pg != 'ml':
        raise ValueError('for analog xyz_to_sop conversion, '
                         'pairing must be "ml"')

    if len(x) == len(y) == 0:
        if not ag:
            return np.array([[z, 0., 0., 1., 0., 0.]])
        else:
            return np.array([[0., 0., z, 0., 0., 1.]])

    if pg != 'ml':
        # ensure we have the same number of poles and zeros, and make copies
        y = np.concatenate((y, np.zeros(max(len(x) - len(y), 0))))
        x = np.concatenate((x, np.zeros(max(len(y) - len(x), 0))))
        c_parts = (max(len(y), len(x)) + 1) // 2

        if len(y) % 2 == 1 and pg == 'nr':
            y = np.concatenate((y, [0.]))
            x = np.concatenate((x, [0.]))
        assert len(y) == len(x)
    else:
        if len(y) < len(x):
            raise ValueError('for analog xyz_to_sop conversion, '
                             'must have len(y)>=len(x)')

        c_parts = (len(y) + 1) // 2

    # Ensure we have complex conjugate pairs
    # (note that _cplxreal only gives us one element of each complex pair):
    x = np.concatenate(_cxr(x))
    y = np.concatenate(_cxr(y))
    if not np.isreal(z):
        raise ValueError('z must be real')
    z = z.real

    if not ag:
        # digital: "worst" is the closest to the unit circle
        iw = lambda a: np.argmin(np.abs(1 - np.abs(a)))
    else:
        # analog: "worst" is the closest to the imaginary axis
        iw = lambda a: np.argmin(np.abs(np.real(a)))

    sop = np.zeros((c_parts, 6))

    # Construct the system, reversing order so the "worst" are last
    for ci in range(c_parts - 1, -1, -1):
        # Select the next "worst" pole
        y1_i = iw(y)
        y1 = y[y1_i]
        y = np.delete(y, y1_i)

        # Pair that pole with a zero

        if np.isreal(y1) and np.isreal(y).sum() == 0:
            # Special case (1): last remaining real pole
            if pg != 'ml':
                x1_i = _closest_real_complex_i(x, y1, 'real')
                x1 = x[x1_i]
                x = np.delete(x, x1_i)
                sop[ci] = _lone_xyzsop([x1, 0], [y1, 0], 1)
            elif len(x) > 0:
                x1_i = _closest_real_complex_i(x, y1, 'real')
                x1 = x[x1_i]
                x = np.delete(x, x1_i)
                sop[ci] = _lone_xyzsop([x1], [y1], 1)
            else:
                sop[ci] = _lone_xyzsop([], [y1], 1)

        elif (len(y) + 1 == len(x)
              and not np.isreal(y1)
              and np.isreal(y).sum() == 1
              and np.isreal(x).sum() == 1):

            # Special case (2): there's one real pole and one real zero
            # left, and an equal number of poles and zeros to pair up.
            # We *must* pair with a complex zero

            x1_i = _closest_real_complex_i(x, y1, 'complex')
            x1 = x[x1_i]
            x = np.delete(x, x1_i)
            sop[ci] = _lone_xyzsop([x1, x1.conj()], [y1, y1.conj()], 1)

        else:
            if np.isreal(y1):
                prealidx = np.flatnonzero(np.isreal(y))
                y2_i = prealidx[iw(y[prealidx])]
                y2 = y[y2_i]
                y = np.delete(y, y2_i)
            else:
                y2 = y1.conj()

            # find closest zero
            if len(x) > 0:
                x1_i = _closest_real_complex_i(x, y1, 'any')
                x1 = x[x1_i]
                x = np.delete(x, x1_i)

                if not np.isreal(x1):
                    sop[ci] = _lone_xyzsop([x1, x1.conj()], [y1, y2], 1)
                else:
                    if len(x) > 0:
                        x2_i = _closest_real_complex_i(x, y1, 'real')
                        x2 = x[x2_i]
                        assert np.isreal(x2)
                        x = np.delete(x, x2_i)
                        sop[ci] = _lone_xyzsop([x1, x2], [y1, y2], 1)
                    else:
                        sop[ci] = _lone_xyzsop([x1], [y1, y2], 1)
            else:
                # no more zeros
                sop[ci] = _lone_xyzsop([], [y1, y2], 1)

    assert len(y) == len(x) == 0  # we've consumed all poles and zeros
    del y, x

    # put gain in first sop
    sop[0][:3] *= z
    return sop

def xyz_to_ptf(x, y, z):
    """
    Parameters
    ----------
    x(array_like): Zeros of the transfer function.
    y(array_like): Poles of the transfer function.
    z(float): System gain.

    Returns
    -------
    g(array_like): Numerator polynomial coefficients.
    h(array_like): Denominator polynomial coefficients.
    """
    x = atleast_1d(x)
    z = atleast_1d(z)
    if len(x.shape) > 1:
        temporary = poly(x[0])
        g = np.empty((x.shape[0], x.shape[1] + 1), temporary.dtype.char)
        if len(z) == 1:
            z = [z[0]] * x.shape[0]
        for i in range(x.shape[0]):
            g[i] = z[i] * poly(x[i])
    else:
        g = z * poly(x)
    h = atleast_1d(poly(y))

    if issubclass(g.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        r = numpy.asarray(x, complex)
        pr = numpy.compress(r.imag > 0, r)
        nr = numpy.conjugate(numpy.compress(r.imag < 0, r))
        if len(pr) == len(nr):
            if numpy.all(numpy.sort_complex(nr) ==
                         numpy.sort_complex(pr)):
                g = g.real.copy()

    if issubclass(h.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        r = numpy.asarray(y, complex)
        pr = numpy.compress(r.imag > 0, r)
        nr = numpy.conjugate(numpy.compress(r.imag < 0, r))
        if len(pr) == len(nr):
            if numpy.all(numpy.sort_complex(nr) ==
                         numpy.sort_complex(pr)):
                h = h.real.copy()

    return g, h


def _closest_real_complex_i(x, y, z):
    """
    Get the next real or complex element based on distance.
    """
    assert z in ('real', 'complex', 'any')
    g = np.argsort(np.abs(x - y))
    if z == 'any':
        return g[0]
    else:
        h = np.isreal(x[g])
        if z == 'complex':
            h = ~h
        return g[np.nonzero(h)[0][0]]