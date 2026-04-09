import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def add_noise(data, noise_level):
    """Add complex Gaussian noise (relative to amplitude)."""
    amplitude = np.abs(data)
    noise_real = noise_level * amplitude * np.random.randn(*data.shape)
    noise_imag = noise_level * amplitude * np.random.randn(*data.shape)
    return data + noise_real + 1j * noise_imag
