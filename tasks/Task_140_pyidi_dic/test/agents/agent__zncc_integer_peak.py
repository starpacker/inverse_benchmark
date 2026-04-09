import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.signal import fftconvolve

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def _zncc_integer_peak(ref_sub, def_region, search_margin):
    """Find integer displacement via ZNCC."""
    n_pix = ref_sub.shape[0] * ref_sub.shape[1]
    ones = np.ones_like(ref_sub)

    ref_zm = ref_sub - ref_sub.mean()
    ref_energy = np.sum(ref_zm**2)
    if ref_energy < 1e-12:
        return 0, 0

    cross = fftconvolve(def_region, ref_zm[::-1, ::-1], mode='valid')
    local_sum = fftconvolve(def_region, ones, mode='valid')
    local_sum2 = fftconvolve(def_region**2, ones, mode='valid')
    local_var = local_sum2 / n_pix - (local_sum / n_pix)**2
    local_var = np.maximum(local_var, 0.0)
    local_energy = local_var * n_pix
    denom = np.sqrt(ref_energy * local_energy)
    denom[denom < 1e-12] = 1e-12
    ncc_map = cross / denom

    peak = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    int_dy = int(peak[0]) - search_margin
    int_dx = int(peak[1]) - search_margin
    return int_dy, int_dx
