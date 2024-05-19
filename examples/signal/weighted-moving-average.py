import proadv as adv
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    """Calculate and plot the weighted moving average of a dataset."""

    # Read data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values

    # Calculate weighted moving average with a period of 10
    weighted_moving_average = adv.statistics.series.weighted_moving_average(main_data, period=30)

    # Plot main data and weighted moving average
    plt.plot(main_data, color='crimson', label='Main Data')
    plt.plot(weighted_moving_average, color='blue', label='Weighted Moving Average')
    plt.legend(loc='upper right')
    plt.xlabel(r'Indexes')
    plt.xlim(8000, 10000)
    plt.title('Weighted Moving Average')
    plt.ylabel(r'Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
