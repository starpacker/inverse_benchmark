import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from scipy.ndimage import gaussian_filter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_complex_object(size=128):
    """Create a complex-valued object with amplitude and phase features."""
    yy, xx = np.mgrid[:size, :size].astype(np.float64)

    # Amplitude: piecewise constant features, then smoothed
    amp = np.ones((size, size), dtype=np.float64) * 0.8
    amp[20:40, 30:60] = 0.3
    amp[60:90, 50:80] = 0.5
    amp[40:70, 20:45] = 0.6
    amp[100:115, 80:110] = 0.25
    amp[15:25, 70:85] = 0.55
    r1 = np.sqrt((yy - 80)**2 + (xx - 30)**2)
    amp[r1 < 15] = 0.35
    r2 = np.sqrt((yy - 30)**2 + (xx - 90)**2)
    amp[r2 < 12] = 0.45
    amp = gaussian_filter(amp, sigma=1.5)

    # Phase: smooth low-frequency variation
    phase = (0.5 * np.sin(2 * np.pi * xx / size) *
             np.cos(2 * np.pi * yy / size))
    phase += 0.8 * np.exp(-((yy - 60)**2 + (xx - 70)**2) / (2 * 25**2))
    phase -= 0.6 * np.exp(-((yy - 40)**2 + (xx - 40)**2) / (2 * 20**2))

    return amp * np.exp(1j * phase)
