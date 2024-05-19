# ProADV - Process Acoustic Doppler Velocimeter

[![GitHub stars](https://img.shields.io/github/stars/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/network)
[![GitHub issues](https://img.shields.io/github/issues/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/issues)
[![GitHub license](https://img.shields.io/github/license/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/proadv.svg)](https://pypi.org/project/proadv/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/proadv.svg)](https://pypi.org/project/proadv/)
[![GitHub contributors](https://img.shields.io/github/contributors/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/graphs/contributors)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/pulls)
[![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/pulls?q=is%3Apr+is%3Aclosed)
[![GitHub last commit](https://img.shields.io/github/last-commit/farzadasgari/proadv)](https://github.com/farzadasgari/proadv/commits/main)


## Streamline Your ADV Data Analysis

**ProADV** is a comprehensive Python package designed to empower researchers and engineers working with acoustic Doppler velocimeter (ADV) data. It offers a comprehensive suite of tools for efficient cleaning, analysis, and visualization of ADV data, streamlining your workflow and extracting valuable insights from your measurements.

### Key Features

* **Despiking and Denoising:** ProADV tackles the challenge of spikes and noise in ADV data, providing a variety of robust algorithms for effective data cleaning. 
    * **Spike Detection:** 
        * **ACC (Acceleration Thresholding):** Identifies spikes based on exceeding a user-defined acceleration threshold.
        * **PST (Phase-Space Thresholding):** Utilizes a combination of velocity and its temporal derivative to detect spikes.
        * **mPST (Modified Phase-Space Thresholding):** An enhanced version of PST with improved sensitivity.
        * **VC (Velocity Correlation):** Detects spikes based on deviations from the correlation between neighboring data points.
        * **KDE (Kernel Density Estimation):** Employs a statistical approach to identify outliers based on the probability density function. 
        * **3d-KDE (Three-dimensional Kernel Density Estimation):** Extends KDE to three dimensions for more robust spike detection in complex data.
        * **m3d-KDE (Modified Three-dimensional Kernel Density Estimation):** Further refines 3d-KDE for enhanced performance.
    * **Replacement Methods:** ProADV offers several options to replace detected spikes with more reliable values:
        * **LVD (Last Valid Data):** Replaces spikes with the last valid data point before the spike.
        *  **MV (Mean Value):** Replaces spikes with the mean value of velocity component. 
        * **LI (Linear Interpolation):** Uses linear interpolation between surrounding points to estimate the missing value.
        * **12PP (12 Points Cubic Polynomial):** Employs a 12-point cubic polynomial to fit a smoother curve and replace spikes.


<div>
   <img src="https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/trivariate-kernel.png" alt="trivariate-kernel" style="width:300px;"/>
   <img src="https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/spectrum.png" alt="trivariate-kernel" style="width:300px;"/>
   <img src="https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/phase-space.png" alt="trivariate-kernel" style="width:300px;"/>
</div>


* **Statistical Analysis:** ProADV equips you with essential statistical tools to characterize your ADV data:
    * **Minimum, Maximum:** Provides the range of measured velocities.
    * **Mean, Median, Mode:** Calculates central tendency measures.
    * **Skewness, Kurtosis:** Analyzes the distribution characteristics of your data.

* **Advanced Analysis:** In addition to cleaning and basic statistics, ProADV offers advanced functionalities for deeper insights:
    * **Moving Average:** Smooths out data fluctuations for better visualization and trend analysis. Provided in simple moving average, exponential moving average, and weighted moving average methods. 
    * **SSA (Singular Spectrum Analysis):** Extracts underlying patterns and trends from time series data.
    * **Kalman Filter:** Implements the Kalman filter algorithm for state estimation and prediction in time series data. 
    * **PR (Pollution Rate) Calculation:** Estimates the level of noise or pollution within the data.
    * **Spectral Analysis:**
        * **PSD (Power Spectral Density):** Analyzes the distribution of energy across different frequencies within the data.
        * **PDF (Probability Density Function):** Provides the probability of encountering specific velocity values.
    * **Normality Test:** Evaluates whether your data follows a normal distribution.
    * **Normalization:** Scales data to a common range for further analysis or visualization.

<div>
   <img src="https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/singular-spectrum.png" alt="singular-spectrum" style="width:300px;"/>
   <img src="https://raw.githubusercontent.com/farzadasgari/proadv/main/examples/plots/kalman.png" alt="kalman-filter" style="width:300px;"/>
</div>

### Installation

There are two convenient ways to install ProADV:

1. **Using pip (recommended):**
   ```bash
   pip install proadv
   ```

2. **From source code:**

   a. Clone the repository:
      ```bash
      git clone https://github.com/farzadasgari/proadv.git
      ```
   b. Navigate to the project directory:
      ```bash
      cd proadv
      ```
   c. Install using setup.py:
      ```bash
      python setup.py install
      ```

### Collaboration

We encourage collaboration and contributions from the community to improve ProADV. Here's how to contribute:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch for your changes.
4. Make your changes and commit them with descriptive messages.
5. Push your changes to your forked repository.
6. Submit a pull request for review and merging.

### References

For further information and in-depth understanding of the algorithms employed in ProADV, refer to the following resources:

1. [Exploring the role of signal pollution rate on the performance of despiking velocity time-series algorithms](https://doi.org/10.1016/j.flowmeasinst.2023.102485)
2. [Unleashing the power of three-dimensional kernel density estimation for Doppler Velocimeter data despiking](https://doi.org/10.1016/j.measurement.2023.114053)


### Acknowledgment
- This project was developed under the supervision of **[Dr. Seyed Hossein Mohaeri](https://khu.ac.ir/cv/1139/Seyed-Hossein-Mohajeri)** and **[Dr. Mojtaba Mehraein](https://khu.ac.ir/cv/279/Mojtaba-Mehraein)**.
- We extend our deepest gratitude to **[Dr. Bimlesh Kumar](https://www.researchgate.net/profile/Bimlesh-Kumar)** and **[Dr. Luis Cea](https://www.researchgate.net/profile/Luis-Cea)** for their invaluable guidance and unwavering support throughout our journey.
- Special thanks to [Parvaneh Yaghoubi](https://github.com/parvanehyaghoubi), [Hossein Abazari](https://github.com/HossA12), [Narges Yaghoubi](https://github.com/nargesyaghoubi), [Mojtaba Karimi](https://github.com/mojikarimi), and [Hiva Yarandi](https://github.com/Hivayrn) for their valuable contributions to this project.


### Contact
For any inquiries, please contact:
- std_farzad.asgari@alumni.khu.ac.ir
- khufarzadasgari@gmail.com


### Links

##### Farzad Asgari
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://farzadasgari.ir/)

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=Rhue_kkAAAAJ&hl=en)

[![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=for-the-badge)](https://www.researchgate.net/profile/Farzad-Asgari)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/farzad-asgari-5a90942b2/)


##### Seyed Hossein Mohajeri
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://khu.ac.ir/cv/1139/Seyed-Hossein-Mohajeri)

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=E8PFUBEAAAAJ&hl=en)

[![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=for-the-badge)](https://www.researchgate.net/profile/Seyed-Mohajeri-2)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](
https://ir.linkedin.com/in/hossein-mohajeri)


##### Mojtaba Mehraein
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://khu.ac.ir/cv/279/Mojtaba-Mehraein)

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=GwT49LIAAAAJ&hl=en)

[![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=for-the-badge)](https://ir.linkedin.com/in/mojtaba-mehraein-002a03238)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](
https://ir.linkedin.com/in/mojtaba-mehraein-002a03238)
