# Modified Three-Dimensional Ketnel Density Estimation (m3d-KDE)

## Introduction

While the 3d-KDE algorithm has shown promise in despiking ADV data, there's an opportunity to enhance its effectiveness by incorporating all three components of velocity as variables in KDE. This approach allows for a more comprehensive analysis of velocity data and potentially improves the accuracy of fluid flow measurements.

## Development of m3d-KDE Algorithm

To address this need, we have developed a modified version of the 3d-KDE algorithm, termed m3d-KDE. This algorithm leverages the correlation between velocity components to identify and remove spikes from ADV data, thereby enabling a more thorough analysis of velocity data and enhancing the accuracy of fluid flow measurements.

## Performance Evaluation

### Comparison with 3d-KDE
We compared the performance of m3d-KDE with the original 3d-KDE algorithm. In Dataset A and Dataset K, m3d-KDE demonstrated superior spike detection capabilities compared to 3d-KDE. Notably, m3d-KDE detected a higher number of spikes in Dataset A and a significantly lower number in Dataset K while preserving the signal's integrity.

### Power Spectral Density (PSD) Analysis
The PSD analysis revealed that m3d-KDE maintained the signal's adherence to Kolmogorov's law, even in highly contaminated datasets like Dataset K. This suggests that m3d-KDE provides a more accurate and reliable approach to despiking ADV data, particularly in challenging environments.

### Sensitivity and Selectivity
Furthermore, m3d-KDE exhibited greater sensitivity in detecting spikes in datasets with low contamination levels and higher selectivity in datasets with high contamination levels compared to 3d-KDE. These observations underscore the versatility and effectiveness of m3d-KDE in mitigating signal contamination across diverse datasets.

## Conclusion
The m3d-KDE algorithm shows promise as a valuable tool for enhancing the accuracy of ADV measurements and improving our understanding of fluid flow dynamics in complex environments. Further research and validation in different flow regimes and environments could provide valuable insights into its effectiveness and applicability.

## Reference
[Unleashing the power of three-dimensional kernel density estimation for Doppler Velocimeter data despiking](https://doi.org/10.1016/j.measurement.2023.114053)
