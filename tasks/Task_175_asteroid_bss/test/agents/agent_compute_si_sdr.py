import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_si_sdr(ref, est):
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e_noise = est - s_target
    return 10.0 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-12))
