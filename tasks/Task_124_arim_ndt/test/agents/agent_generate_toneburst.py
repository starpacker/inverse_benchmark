import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_toneburst(fc, n_cycles, fs):
    """Generate a Hanning-windowed tone burst."""
    duration = n_cycles / fc
    n_pts = int(np.ceil(duration * fs))
    t_burst = np.arange(n_pts) / fs
    burst = np.sin(2 * np.pi * fc * t_burst) * np.hanning(n_pts)
    return t_burst, burst
