from scipy.signal import welch
from scipy.signal.windows import hann


def power_spectra(data, sampling_frequency, overlap=50, block=2048, onesided=True):
    win=hann(block, True)
    fxu, fyu = welch(data, sampling_frequency, window=win, noverlap=overlap, nfft=block, return_onesided=onesided)
    return fxu, fyu
    
