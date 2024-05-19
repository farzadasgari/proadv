import proadv as adv
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    """Calculate and plot the exponential moving average of a dataset."""

    # Read data from CSV file
    df = read_csv('../../dataset/second.csv')
    main_data = df.iloc[:, 0].values

    # Calculate exponential moving average with a alpha value of 0.08
    exponential_moving_average = adv.statistics.series.exponential_moving_average(main_data, alpha=0.08)

    # Plot main data and simple moving average
    plt.plot(main_data, color='crimson', label='Main Data')
    plt.plot(exponential_moving_average, color='blue', label='Exponential Moving Average')
    plt.legend(loc='upper right')
    plt.xlabel(r'Indexes')
    plt.xlim(6000, 8000)
    plt.title('Exponential Moving Average')
    plt.ylabel(r'Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
