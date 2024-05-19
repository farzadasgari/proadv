import numpy as np


def _relative_scale(zero, poles):
    """
    Return the relative scale of the transfer function from zero and pole
    """
    scale = np.copy(poles).shape[0] - np.copy(zero).shape[0]
    if scale < 0:
        raise ValueError("Improper transfer function. Must have at least as many poles as zeros.")
    else:
        return scale


def _catenate(array, fpc):
    """
     To join multiple presentations.
    """
    catenate_array = np.concatenate((array + np.sqrt(array ** 2 - fpc ** 2),
                                     array - np.sqrt(array ** 2 - fpc ** 2)))
    return catenate_array


def low_to_stop(zero, poles, system, pc=1.0, pw=1.0):
    """
    This function converts a LP(low-pass) filter prototype to a BS(band-stop) filter.

    Parameters
    ------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float): System gain of the analog filter transfer function.
    pc (float): Desired stopband center, as angular frequency.
    pw (float): Desired stopband width, as angular frequency.

    Returns
    ------
    zerob (array_like): Zeros of the transformed BS(band-stop) filter transfer function.
    polesb (array_like): Poles of the transformed BS(band-stop) filter transfer function.
    systemb (float): System gain of the transformed BS(band-stop) filter.

    Examples
    ------
    >>> from proadv.statistics.signal.butter import low_to_stop
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
    zero = np.atleast_1d(zero)
    poles = np.atleast_1d(poles)
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

    zerob = np.append(zerob, np.full(scale, +1j * fpc))
    zerob = np.append(zerob, np.full(scale, -1j * fpc))

    systemb = system * np.real(np.prod(-zero) / np.prod(-poles))

    return zerob, polesb, systemb


def low_to_band(zero, poles, system, pc=1.0, pw=1.0):
    """
    This function converts a LP(low-pass) filter prototype to a BP(band-pass) filter.

    Parameters
    ------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float): System gain of the analog filter transfer function.
    pc (float): Desired passband center, as angular frequency.
    pw (float): Desired passband width, as angular frequency.

    Returns

    zerob (array_like): Zeros of the transformed BP(band-pass) filter transfer function.
    polesb (array_like): Poles of the transformed BP(band-pass) filter transfer function.
    systemb (float): System gain of the transformed BP(band-pass) filter.
    ------

    Examples
    ------
    >>> from proadv.statistics.signal.butter import low_to_band
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
    zero = np.atleast_1d(zero)
    poles = np.atleast_1d(poles)
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

    zerob = np.append(zerob, np.zeros(scale))

    # Cancel out gain change from frequency scaling
    systemb = system * fpw ** scale

    return zerob, polesb, systemb


def low_to_high(zero, poles, system, dc=1.0):
    """
    This function converts a low pass filter prototype to another frequency.

    Parameters
    ------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float): System gain of the analog filter transfer function.
    dc (float): Desired cutoff, as angular frequency.

    Returns
    ------
    zeroh (array_like): Zeros of the transformed LP(low-pass) filter transfer function.
    polesh (array_like): Poles of the transformed LP(low-pass) filter transfer function.
    systemh (float): System gain of the transformed LP(low-pass) filter.

    """
    zero = np.atleast_1d(zero)
    poles = np.atleast_1d(poles)
    fdc = float(dc)

    scale = _relative_scale(zero, poles)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    zeroh = fdc / zero
    polesh = fdc / poles

    # If lowpass had zeros at infinity, inverting moves them to origin.
    zeroh = np.append(zeroh, np.zeros(scale))

    # Cancel out gain change caused by inversion
    systemh = system * np.real(np.prod(-zero) / np.prod(-poles))

    return zeroh, polesh, systemh


def bilinear(zero, poles, system, sr):
    """
    This function returns a digital IIR filter using a bilinear conversion from an analog one.

    Parameters
    ----------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float): System gain of the analog filter transfer function.
    sr (float): Sample rate, as ordinary frequency.

    Returns
    -------
    zerob (array_like): Zeros of the transformed digital filter transfer function.
    polesb (array_like): Poles of the transformed digital filter transfer function.
    systemb (float): System gain of the transformed digital filter.

    """
    zero = np.atleast_1d(zero)
    poles = np.atleast_1d(poles)

    fsr = _valid_g(sr, an=False)

    scale = _relative_scale(zero, poles)

    fsr2 = 2.0 * fsr

    # Bilinear transform the poles and zeros
    lf = lambda x, y: (x + y) / (x - y)
    zerob = lf(fsr2, zero)
    polesb = lf(fsr2, poles)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    zerob = np.append(zerob, -np.ones(scale))

    # Compensate for gain change
    systemb = system * np.real(np.prod(fsr2 - zero) / np.prod(fsr2 - poles))

    return zerob, polesb, systemb


def low_to_low(zero, poles, system, dc=1.0):
    """
    This function converts a LP(low-pass) filter prototype to another frequency.

    Parameters
    ----------
    zero (array_like): Zeros of the analog filter transfer function.
    poles (array_like): Poles of the analog filter transfer function.
    system (float): System gain of the analog filter transfer function.
    dc (float): Desired cutoff, as angular frequency.

    Returns
    -------
    zerol (array_like): Zeros of the transformed LP(low-pass) filter transfer function.
    polesl (array_like): Poles of the transformed LP(low-pass) filter transfer function.
    systeml (float): System gain of the transformed LP(low-pass) filter.

    Examples
    ------
    >>> from proadv.statistics.signal.butter import low_to_low
    >>> zero = [8, 3]
    >>> poles = [6, 14]
    >>> system = 0.7
    >>> dc = 0.5
    >>> low_to_low(zero, poles, system, dc)
    (array([4. , 1.5]), array([3., 7.]), 0.7)

    """
    zero = np.atleast_1d(zero)
    poles = np.atleast_1d(poles)
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
    x = np.array([])
    y = np.arange(-number + 1, number, 2)
    # Middle value is 0 to ensure an exactly real pole
    z = -np.exp(1j * np.pi * y / (2 * number))
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
    x = np.atleast_1d(x)
    z = np.atleast_1d(z)
    if len(x.shape) > 1:
        temporary = np.poly(x[0])
        g = np.empty((x.shape[0], x.shape[1] + 1), temporary.dtype.char)
        if len(z) == 1:
            z = [z[0]] * x.shape[0]
        for i in range(x.shape[0]):
            g[i] = z[i] * np.poly(x[i])
    else:
        g = z * np.poly(x)
    h = np.atleast_1d(np.poly(y))

    if issubclass(g.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        r = np.asarray(x, complex)
        pr = np.compress(r.imag > 0, r)
        nr = np.conjugate(np.compress(r.imag < 0, r))
        if len(pr) == len(nr):
            if np.all(np.sort_complex(nr) ==
                      np.sort_complex(pr)):
                g = g.real.copy()

    if issubclass(h.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        r = np.asarray(y, complex)
        pr = np.compress(r.imag > 0, r)
        nr = np.conjugate(np.compress(r.imag < 0, r))
        if len(pr) == len(nr):
            if np.all(np.sort_complex(nr) ==
                      np.sort_complex(pr)):
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


def _cxr(x, me=None):
    x = np.atleast_1d(x)
    if x.size == 0:
        return x, x
    elif x.ndim != 1:
        raise ValueError('_cxr only accepts 1-D input')

    if me is None:
        # Get tolerance from dtype of input
        me = 100 * np.finfo((1.0 * x).dtype).eps

    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    x = x[np.lexsort((abs(x.imag), x.real))]

    # Split reals from conjugate pairs
    ri = np.abs(x.imag) <= me * np.abs(x)
    xr = x[ri].real

    if np.array(xr).shape[0] == np.array(x).shape[0]:
        # Input is entirely real
        return np.array([]), xr

    # Split positive and negative halves of conjugates
    x = x[~ri]
    xp = x[x.imag > 0]
    xn = x[x.imag < 0]

    if np.array(xp).shape[0] != np.array(xn).shape[0]:
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    sr = np.diff(xp.real) <= me * abs(xp[:-1])
    variety = np.diff(np.concatenate(([0], sr, [0])))
    rst = np.nonzero(variety > 0)[0]
    rss = np.nonzero(variety < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(np.array(rst).shape[0]):
        st = rst[i]
        sp = rss[i] + 1
        for chunk in (xp[st:sp], xn[st:sp]):
            chunk[...] = chunk[np.lexsort([np.abs(chunk.imag)])]

    # Check that negatives match positives
    if any(abs(xp - xn.conj()) > me * abs(xn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    xc = (xp + xn.conj()) / 2

    return xc, xr


def _lone_xyzsop(x, y, z):
    """
    Create one second-order part from up to two zeros and poles.
    """
    sop = np.zeros(6)
    g, h = xyz_to_ptf(x, y, z)
    sop[3 - len(g):3] = g
    sop[6 - len(h):6] = h
    return sop


def _valid_g(g, an=True):
    """
    Check if the given sampling frequency is a scalar and raises an exception
    otherwise. If allow_none is False, also raises an exception for none
    sampling rates. Returns the sampling frequency as float or none if the
    input is none.
    """
    if g is None:
        if not an:
            raise ValueError("Sampling frequency can not be none.")
    else:  # should be floated
        if not np.isscalar(g):
            raise ValueError("Sampling frequency fs must be a single scalar.")
        g = float(g)
    return g


def butterworth(q, w, btype='bp', ag=False, output='nd', sr=None):
    """
    Butterworth digital and analog filter design.

    Parameters
    ----------
    q(int) :The order of the filter.
    w(array_like): The critical frequency or frequencies.
    btype(optional): The type of filter.
        btype types:
        - lp(lowpass)
        - hp(highpass)
        - bp(bandpass)
        - bs(bandstop)
    ag(bool, optional): When False, return a digital filter, otherwise return an analog filter.
    output(str,optional): Specifies the output type of the code.
        output types:
        - nd
        - zps
        - sop
    sr(float, optional): The sampling frequency of the digital system.

    Returns
    -------

    """
    sr = _valid_g(sr, an=True)
    btype, output = (x.lower() for x in (btype, output))
    w = np.asarray(w)
    if sr is not None:
        if ag:
            raise ValueError("fs cannot be specified for an analog filter")
        w = 2 * w / sr

    if np.any(w <= 0):
        raise ValueError("filter critical frequencies must be greater than 0")

    if w.size > 1 and not w[0] < w[1]:
        raise ValueError("w[0] must be less than w[1]")

    if output not in ['nd', 'zps', 'sop']:
        raise ValueError("'%s' is not a valid output form." % output)

    # Get analog lowpass prototype
    zero, poles, system = buttap(q)

    # Pre-warp frequencies for digital filter design
    if not ag:
        if np.any(w <= 0) or np.any(w >= 1):
            if sr is not None:
                raise ValueError("Digital filter critical frequencies must "
                                 f"be 0 < w < sr/2 (sr={sr} -> sr/2={sr / 2})")
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < w < 1")
        sr = 2.0
        dc = 2 * sr * np.tan(np.pi * w / sr)
    else:
        dc = w

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lp', 'hp'):
        if np.size(w) != 1:
            raise ValueError('Must specify a single critical frequency w '
                             'for lowpass or highpass filter')

        if btype == 'lp':
            zero, poles, system = low_to_low(zero, poles, system, dc=dc)
        elif btype == 'hp':
            zero, poles, system = low_to_high(zero, poles, system, dc=dc)
    elif btype in ('bp', 'bs'):
        try:
            pw = dc[1] - dc[0]
            g = np.sqrt(dc[0] * dc[1])
        except IndexError as e:
            raise ValueError('w must specify start and stop frequencies for '
                             'bandpass or bandstop filter') from e

        if btype == 'bp':
            zero, poles, system = low_to_band(zero, poles, system, pc=g, pw=pw)
        elif btype == 'bs':
            zero, poles, system = low_to_stop(zero, poles, system, pc=g, pw=pw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not ag:
        zero, poles, system = bilinear(zero, poles, system, sr=sr)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zps':
        return zero, poles, system
    elif output == 'nd':
        return xyz_to_ptf(zero, poles, system)
    elif output == 'sop':
        return xyz_to_sop(zero, poles, system, ag=ag)
