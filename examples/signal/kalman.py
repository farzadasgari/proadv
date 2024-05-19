import proadv as adv
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    """Calculate and plot the Kalman filter of a given dataset."""

    # Read data from CSV file
    df = read_csv('../../dataset/first.csv')
    main_data = df.iloc[:, 0].values

    initial_state = np.array([[1]])
    initial_covariance = np.array([[1]])
    process_noise = np.array([[0.01]])
    measurement_noise = np.array([[15]])

    # Calculate Kalman filter
    kalman = adv.statistics.series.kalman_filter(main_data,
                                                 initial_state,
                                                 initial_covariance,
                                                 process_noise,
                                                 measurement_noise)

    plt.plot(np.squeeze(main_data), color='crimson', label='Main Data')
    plt.plot(np.squeeze(kalman), color='black', label='Kalman')
    plt.legend(loc='upper right')
    plt.title('Kalman Filter')
    plt.xlabel(r'Indexes')
    plt.ylabel(r'Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
