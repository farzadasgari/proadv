from scipy.signal import welch
from scipy.signal.windows import hann


def power_spectra(data, sampling_frequency, overlap=50, block=2048, onesided=True):
    """
    Compute the power spectra of the input data.

    Parameters
    ------
    data (array_like): Input data array.
    sampling_frequency (float): Sampling frequency of the input data.
    overlap (int, optional): Percentage overlap between blocks. Default is 50.
    block (int, optional): Length of each block for computing the spectra. Default is 2048.
    onesided (bool, optional): If True, return one-sided spectrum. Default is True.

    Returns
    ------
    fxu (array_like): Array of frequencies.
    fyu (array_like): Power spectral density of the input data.
    """

    # Define Hann window
    win=hann(block, True)

    # Compute power spectral density
    fxu, fyu = welch(data, sampling_frequency, window=win, noverlap=overlap, nfft=block, return_onesided=onesided)
    
    return fxu, fyu
