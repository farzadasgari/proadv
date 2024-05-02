# ProADV (Process Acoustic Doppler Velocimeter)

[![GitHub stars](https://img.shields.io/github/stars/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/network)
[![GitHub issues](https://img.shields.io/github/issues/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/issues)
[![GitHub license](https://img.shields.io/github/license/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/proadv.svg)](https://badge.fury.io/py/proadv)
[![conda](https://img.shields.io/conda/vn/conda-forge/proadv.svg)](https://anaconda.org/conda-forge/proadv)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/proadv)](https://pypi.org/project/proadv/)
[![GitHub contributors](https://img.shields.io/github/contributors/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/graphs/contributors)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/pulls)
[![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/pulls?q=is%3Apr+is%3Aclosed)
[![GitHub last commit](https://img.shields.io/github/last-commit/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/commits/main)


ProAdv is a Python package designed for post-processing acoustic Doppler velocimeter datasets, offering denoising and despiking methods.

## Spike Detection Methods
- ACC (Acceleration Thresholding)
- PST (Phase-Space Thresholding)
- mPST (Modified Phase-Space Thresholding)
- VC (Velocity Correlation)
- KDE (Kernel Density Estimation)
- 3d-KDE (Three-dimensional Kernel Density Estimation)
- m3d-KDE (Modified Three-dimensional Kernel Density Estimation)

## Replacement Methods
- LGV (Last Good Value)
- LI (Linear Interpolation)
- 12PP (12 Points Cubic Polynomial)

## Statistical Functions
ProAdv provides statistical functions such as:
- Minimum
- Maximum
- Mean
- Median
- Skewness
- Kurtosis

## Analysis
In addition to despiking, ProADV offers analysis including:
- Moving Average
- SSA (Singular Spectrum Analysis)
- Calculation of PR (Pollution Rate)
- PSD (Power Spectral Density)
- PDF (Probability Density Function)
- Normality Test
- Normalization 

## Installation
You can install ProAdv using pip:

```bash
pip install proadv
```

Or using conda:

```bash
conda install -c conda-forge proadv
```

Alternatively, you can install it from the source code via `setup.py`:

```bash
git clone https://github.com/farzadasgari/proadv.git
cd proadv
python setup.py install
```

## Collaboration
We welcome collaboration and contributions from the community. To contribute to ProAdv, follow these steps:

1. **Fork** the repository by clicking the "Fork" button on the top right corner of this page.
2. **Clone** your forked repository to your local machine:
   ```bash
   git clone https://github.com/your-username/proadv.git
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and commit them with descriptive commit messages.
5. **Push** your changes to your forked repository:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Finally, submit a **pull request** to the `main` branch of the original repository for review.

## References
1. [Exploring the role of signal pollution rate on the performance of despiking velocity time-series algorithms](https://doi.org/10.1016/j.flowmeasinst.2023.102485)
2. [Unleashing the power of three-dimensional kernel density estimation for Doppler Velocimeter data despiking](https://doi.org/10.1016/j.measurement.2023.114053)

## Contact
For any inquiries, please contact:
- Email: std_farzad.asgari@gmail.com
- Email: khufarzadasgari@gmail.com
