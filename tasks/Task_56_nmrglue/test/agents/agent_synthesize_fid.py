import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def synthesize_fid(peaks, n_f1, n_f2, sw_f1, sw_f2):
    """
    Synthesize a 2D FID from Lorentzian peaks.
    """
    dt1 = 1.0 / sw_f1
    dt2 = 1.0 / sw_f2
    t1 = np.arange(n_f1) * dt1
    t2 = np.arange(n_f2) * dt2

    fid = np.zeros((n_f1, n_f2), dtype=complex)
    for p in peaks:
        decay_f1 = np.exp(-np.pi * p["lw_f1"] * t1)
        osc_f1 = np.exp(1j * 2 * np.pi * p["freq_f1"] * t1)
        decay_f2 = np.exp(-np.pi * p["lw_f2"] * t2)
        osc_f2 = np.exp(1j * 2 * np.pi * p["freq_f2"] * t2)
        sig_f1 = p["amplitude"] * np.exp(1j * p["phase"]) * decay_f1 * osc_f1
        sig_f2 = decay_f2 * osc_f2
        fid += np.outer(sig_f1, sig_f2)

    return fid
