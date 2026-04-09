import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_speckle_image(height, width, n_speckles=15000,
                           speckle_sigma=2.5, seed=42):
    """Generate a dense synthetic speckle pattern for DIC."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width), dtype=np.float64)

    ys = rng.uniform(0, height, n_speckles)
    xs = rng.uniform(0, width, n_speckles)
    intensities = rng.uniform(0.3, 1.0, n_speckles)

    r = int(4 * speckle_sigma) + 1
    for y0, x0, amp in zip(ys, xs, intensities):
        y_lo = max(0, int(y0) - r)
        y_hi = min(height, int(y0) + r + 1)
        x_lo = max(0, int(x0) - r)
        x_hi = min(width, int(x0) + r + 1)
        yy = np.arange(y_lo, y_hi)[:, None]
        xx = np.arange(x_lo, x_hi)[None, :]
        gauss = amp * np.exp(-((yy - y0)**2 + (xx - x0)**2) /
                              (2 * speckle_sigma**2))
        img[y_lo:y_hi, x_lo:x_hi] += gauss

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img
