import proadv as adv
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    """Calculate and plot the Singular Spectrum of a given dataset."""

    # Read data from CSV file
    df = read_csv('../../dataset/first.csv')
    main_data = df.iloc[:, 1].values

    # Sampling frequency
    sampling_frequencty = 100

    # Calculate Singular Spectrum Analysis
    singular_spectrum = adv.statistics.series.ssa(main_data, sampling_frequencty, f=1)

    plt.plot(main_data, color='crimson', label='Main Data')
    plt.plot(singular_spectrum, color='black', label='Singular Spectrum')
    plt.legend(loc='upper right')
    plt.title('Singular Spectrum')
    plt.xlabel(r'Indexes')
    plt.ylabel(r'Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
