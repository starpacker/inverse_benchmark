import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def align_gains(g_cal: np.ndarray, g_true: np.ndarray, ref_ant: int) -> np.ndarray:
    """Align calibrated gains to true gains by removing global phase offset."""
    ratio = g_true / np.where(np.abs(g_cal) > 1e-15, g_cal, 1e-15)
    mask = np.ones(g_cal.shape[0], dtype=bool)
    mask[ref_ant] = False
    ratio_excl = ratio[mask]
    phase_offset = np.angle(np.mean(ratio_excl))
    g_aligned = g_cal * np.exp(1j * phase_offset)
    return g_aligned
