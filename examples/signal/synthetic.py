import proadv as adv
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Generate and plot synthetic polluted data."""
    np.random.seed(0)

    # Generate random data
    data = np.random.randint(0, 50, 500)

    # Generate synthetic polluted data
    artificial_pollution = adv.statistics.signal.synthetic.synthetic_noise(data, percent=10)

    # Plot the synthetic polluted data and the original data
    plt.plot(artificial_pollution, color='crimson', label='Synthetic Pollution')
    plt.plot(data, label="Main Data")
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
