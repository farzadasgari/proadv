# Power Spectra 

This function compute the `Power Spectra` of the input data and it returns two matters:

- **fxu (array_like)**: Array of frequencies.
- **fyu (array_like)**: Power spectral density of the input data.

There are five parameters in this function:

- **data (array_like)**: Input data array.
- **sampling_frequency (float)**: Sampling frequency of the input data.
- **overlap (int, optional)**: Percentage overlap between blocks. Default is 50.
- **block (int, optional)**: Length of each block for computing the spectra. Default is 2048.
- **onesided (bool, optional)**: If True, return one-sided spectrum. Default is True.

