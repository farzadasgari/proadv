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