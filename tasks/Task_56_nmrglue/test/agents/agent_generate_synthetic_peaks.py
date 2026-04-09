import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_synthetic_peaks(n_peaks, sw_f1, sw_f2, seed=42):
    """
    Generate random peak parameters for a 2D NMR spectrum.
    """
    rng = np.random.default_rng(seed)
    peaks = []
    for i in range(n_peaks):
        peaks.append({
            "freq_f1": rng.uniform(0.15, 0.85) * sw_f1,
            "freq_f2": rng.uniform(0.15, 0.85) * sw_f2,
            "lw_f1": rng.uniform(10, 50),
            "lw_f2": rng.uniform(15, 80),
            "amplitude": rng.uniform(0.5, 2.0),
            "phase": rng.uniform(-0.1, 0.1),
        })
    return peaks
