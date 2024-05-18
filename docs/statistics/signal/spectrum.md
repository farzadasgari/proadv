# Power Spectral Density (PSD) Analysis

## Introduction

Power Spectral Density (PSD) analysis is a powerful technique used in signal processing and data analysis to understand the frequency content or spectral characteristics of a signal. It provides valuable insights into the distribution of signal power across different frequency components.

## Power Spectra and Frequency Analysis

In frequency analysis, a signal can be decomposed into its constituent frequencies using techniques like Fourier analysis. Power spectra represent the distribution of signal power across these frequencies. By examining the PSD of a signal, one can identify dominant frequency components and their relative strengths.

## Kolmogorov's -5/3 Law

Kolmogorov's -5/3 law, also known as the Kolmogorov spectrum, is a fundamental concept in turbulence theory. It describes the behavior of turbulent flows and is characterized by a power-law relationship between the PSD of velocity fluctuations and the frequency of turbulent eddies. According to this law, in the inertial subrange of fully developed turbulence, the PSD follows a power-law decay with an exponent of -5/3.

## Usage

The `power_spectra` function provided by the ProADV, calculates the Power Spectral Density (PSD) of a given dataset. This function accepts the dataset and the sampling frequency as input and returns the PSD along with the corresponding frequencies. It employs advanced signal processing techniques to compute the PSD efficiently.

```python
import proadv as adv
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    """Calculate and plot the power spectral density (PSD) of a given dataset."""

    # Read data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values

    # Sampling frequency
    sampling_frequencty = 100

    # Calculate power spectral density (PSD) and frequencies
    psdx, freqs = adv.statistics.signal.spectrum.power_spectra(main_data, sampling_frequencty)

    # Define slope for Kolmogorov -5/3 law
    slope = - 5 / 3

    # Calculate coordinates for plotting Kolmogorov -5/3 law
    ymin1, ymax1 = np.log([freqs.min(), freqs.max()])
    ymid1 = (ymin1 + ymax1) / 2
    x1, x2 = 0, np.log(psdx.max())
    xmid1 = (x1 + x2) / 2
    y1 = slope * (x1 - xmid1) + ymid1
    y2 = slope * (x2 - xmid1) + ymid1

    # Plot PSD and Kolmogorov -5/3 law
    plt.plot(psdx, freqs, lw=0.9, color='crimson', label='Power Spectra')
    plt.plot(np.exp([x1, x2]), np.exp([y1, y2]), '-.k', label='Kolmogorov -5/3 law', lw=0.6)
    plt.legend(loc='lower left')
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'PSD (cm$^2$ / s)')
    plt.loglog()
    plt.show()


if __name__ == '__main__':
    main()
```

![spectrum](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/spectrum.png)

## Conclusion

Power Spectral Density (PSD) analysis, along with concepts like Kolmogorov's law, is essential for understanding the frequency characteristics of signals and phenomena ranging from turbulence in fluid dynamics to fluctuations in financial markets. By leveraging the `power_spectra` function, researchers and practitioners can gain valuable insights into the spectral properties of their data, aiding in various scientific and engineering applications.
