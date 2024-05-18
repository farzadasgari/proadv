import proadv as adv
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values
    sampling_frequencty = 100
    psdx, freqs = adv.statistics.signal.spectrum.power_spectra(main_data, sampling_frequencty)
    slope = - 5 / 3

    ymin1, ymax1 = np.log([freqs.min(), freqs.max()])
    ymid1 = (ymin1 + ymax1) / 2
    x1, x2 = 0, np.log(psdx.max())
    xmid1 = (x1 + x2) / 2
    y1 = slope * (x1 - xmid1) + ymid1
    y2 = slope * (x2 - xmid1) + ymid1

    plt.figure(num='Spectrum', figsize=(8, 6))
    plt.plot(psdx, freqs, lw=0.9, color='crimson', label='Power Spectra')
    plt.plot(np.exp([x1, x2]), np.exp([y1, y2]), '-.k', label='Kolmogorov -5/3 law', lw=0.6)
    plt.legend(loc='lower left')
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'PSD (cm$^2$ / s)')
    plt.loglog()
    plt.show()


if __name__ == '__main__':
    main()