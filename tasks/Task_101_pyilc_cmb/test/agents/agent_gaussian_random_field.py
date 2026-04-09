import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_101_pyilc_cmb"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def gaussian_random_field(N, power_law_index=-2.0, rms=1.0, seed=None):
    """Generate a 2D Gaussian random field with power-law power spectrum."""
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0  # avoid division by zero
    power = K ** power_law_index
    power[0, 0] = 0.0  # zero mean
    phases = np.random.uniform(0, 2 * np.pi, (N, N))
    amplitudes = np.sqrt(power) * np.exp(1j * phases)
    field = np.fft.ifft2(amplitudes).real
    field = field / field.std() * rms
    return field
