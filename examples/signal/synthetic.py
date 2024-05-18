import proadv as adv
import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(0)
    data = np.random.randint(0, 50, 500)
    artificial_pollution = adv.statistics.signal.synthetic.synthetic_noise(data, percent=10)

    plt.plot(artificial_pollution, color='crimson', label='Synthetic Pollution')
    plt.plot(data, label="Main Data")
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
