# Synthetic Pollution

## Purpose
The `synthetic_noise` function in ProADV generates synthetic noisy data by adding artificial pollution to the input data. This function is useful for simulating real-world scenarios where data might be corrupted or contaminated by noise.

## Function Signature
```python
def synthetic_noise(data, percent):
    """
    Generate synthetic noisy data based on the input data.

    Parameters
    ------
    data (array_like): The original data.
    percent (float): The percentage of data points to perturb.

    Returns
    ------
    synthetic_polluted_data (np.ndarray): Synthetic data with added noise.
    """
```

## Usage
```python
import proadv as adv
import numpy as np

# Generate random data
data = np.random.randint(0, 50, 500)

# Generate synthetic polluted data
artificial_pollution = adv.statistics.signal.synthetic.synthetic_noise(data, percent=10)

# Plot the synthetic polluted data and the original data
import matplotlib.pyplot as plt
plt.plot(artificial_pollution, color='crimson', label='Synthetic Pollution')
plt.plot(data, label="Main Data")
plt.legend(loc='upper right')
plt.show()
```

![acceleration](https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/synthetic.png)

In this example, we first generate random data using NumPy's `randint` function. Real measured dataset could be used here. Then, we use the `synthetic_noise` function to add artificial pollution to the data with a noise level of 10%. Finally, we plot both the original data and the synthetic polluted data for visualization.

## Importance for Researchers
The ability to generate synthetic noisy data is an important feature for researchers in various fields, including data science, machine learning, and signal processing. It allows researchers to create realistic datasets for testing algorithms, evaluating models, and studying the impact of noise on data analysis techniques. By simulating real-world scenarios, researchers can gain insights into the robustness and performance of their methods under different conditions.

## Notes
- The `percent` parameter controls the level of pollution to be added to the data. Adjusting this parameter allows for varying degrees of noise in the generated synthetic data.
- The function utilizes NumPy for generating random indexes and adding noise to the data.
