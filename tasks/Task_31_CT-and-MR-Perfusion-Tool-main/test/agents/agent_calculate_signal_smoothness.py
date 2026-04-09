import numpy as np

import warnings

warnings.filterwarnings("ignore")

def calculate_signal_smoothness(signal):
    if len(signal) < 3: return float('inf')
    rnge = np.max(signal) - np.min(signal)
    if rnge == 0: return 0.0
    norm_sig = (signal - np.min(signal)) / rnge
    second_deriv = np.diff(norm_sig, n=2)
    return np.sum(second_deriv ** 2) / len(signal)
