import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import uniform_filter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_true_gains(n_ant: int, n_freq: int, n_time: int, ref_ant: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate true complex gains per antenna, per frequency, per time.
    Shape: (n_ant, n_freq, n_time)
    """
    amplitudes = np.zeros((n_ant, n_freq, n_time))
    phases_deg = np.zeros((n_ant, n_freq, n_time))

    for a in range(n_ant):
        amp_base = rng.uniform(0.8, 1.2)
        phase_base = rng.uniform(-30.0, 30.0)
        amp_var = rng.normal(0, 0.03, size=(n_freq, n_time))
        phase_var = rng.normal(0, 3.0, size=(n_freq, n_time))
        amp_var = uniform_filter(amp_var, size=3)
        phase_var = uniform_filter(phase_var, size=3)
        amplitudes[a] = amp_base + amp_var
        phases_deg[a] = phase_base + phase_var

    amplitudes[ref_ant] = 1.0
    phases_deg[ref_ant] = 0.0

    phases_rad = np.deg2rad(phases_deg)
    gains = amplitudes * np.exp(1j * phases_rad)
    return gains
