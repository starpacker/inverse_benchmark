import numpy as np

import matplotlib

matplotlib.use("Agg")

def make_ground_truth_phase(shape):
    """
    Create a 2-D phase map with structured features.
    """
    H, W = shape
    y, x = np.mgrid[:H, :W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0

    phase = np.zeros((H, W), dtype=np.float64)
    rng = np.random.RandomState(42)

    n_peaks = 12
    for _ in range(n_peaks):
        px = rng.uniform(W * 0.15, W * 0.85)
        py = rng.uniform(H * 0.15, H * 0.85)
        sigma = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.15, 0.35)
        phase += amp * np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma**2))

    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    phase += 0.25 * np.exp(-((r - min(H, W) * 0.25) / 5.0)**2)

    phase = phase / (phase.max() + 1e-12) * 0.5
    return phase
