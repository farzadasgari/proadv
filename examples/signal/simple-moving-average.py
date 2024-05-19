import proadv as adv
import matplotlib.pyplot as plt
from pandas import read_csv


def main():
    df = read_csv('../../dataset/first.csv')
    main_data = df.iloc[:, 1].values
    simple_moving_average = adv.statistics.series.moving_average(main_data, window_size=30)
    plt.plot(main_data, color='crimson', label='Main Data')
    plt.plot(simple_moving_average, color='blue', label='Simple Moving Average')
    plt.legend(loc='upper right')
    plt.xlabel(r'Indexes')
    plt.xlim(2000, 4000)
    plt.ylabel(r'Velocity (cm/s)')
    plt.show()


if __name__ == '__main__':
    main()
