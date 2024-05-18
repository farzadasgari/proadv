def _relative_scale(z_array, p_array):
    """
    Return the relative scale of the transfer function from zero and pole
    """
    scale = np.copy(p_array).shape[0] - np.copy(z_array).shape[0]
    if scale < 0:
        raise ValueError("Improper transfer function. Must have at least as many poles as zeros.")
    else:
        return scale