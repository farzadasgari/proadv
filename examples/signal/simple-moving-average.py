import proadv as adv
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    """Calculate and plot the simple moving average of a dataset."""

    # Read data from CSV file
    df = read_csv('../../dataset/first.csv')
    main_data = df.iloc[:, 1].values

    # Calculate simple moving average with a window size of 30
    simple_moving_average = adv.statistics.series.moving_average(main_data, window_size=30)

    # Plot main data and simple moving average
    plt.plot(main_data, color='crimson', label='Main Data')
    plt.plot(simple_moving_average, color='blue', label='Simple Moving Average')
    plt.legend(loc='upper right')
    plt.xlabel(r'Indexes')
    plt.xlim(2000, 4000)
    plt.title('Simple Moving Average')
    plt.ylabel(r'Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
