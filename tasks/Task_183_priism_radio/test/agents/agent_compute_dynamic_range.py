import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_dynamic_range(image, source_mask):
    """Ratio of peak signal to rms in background region."""
    bg = image[~source_mask]
    rms = np.sqrt(np.mean(bg ** 2)) if len(bg) > 0 else 1e-15
    if rms < 1e-15:
        rms = 1e-15
    return float(image.max() / rms)
