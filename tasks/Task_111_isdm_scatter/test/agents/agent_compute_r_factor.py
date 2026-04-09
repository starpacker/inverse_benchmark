import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def compute_r_factor(measured_magnitude, recon):
    """Compute R-factor (Fourier-space error) to evaluate reconstruction quality."""
    F_recon = np.fft.fft2(recon)
    recon_mag = np.abs(F_recon)
    r_factor = np.sum(np.abs(measured_magnitude - recon_mag)) / np.sum(measured_magnitude + 1e-12)
    return r_factor
