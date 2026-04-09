import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_undersampling_mask(shape, acceleration=4, center_fraction=0.08, seed=42):
    """
    Create a random Cartesian undersampling mask.
    """
    rng = np.random.RandomState(seed)
    ny, nx = shape
    mask = np.zeros(shape, dtype=np.float64)

    num_center = int(center_fraction * ny)
    center_start = (ny - num_center) // 2
    mask[center_start:center_start + num_center, :] = 1.0

    num_total_lines = ny // acceleration
    num_random_lines = max(num_total_lines - num_center, 0)

    available = list(set(range(ny)) - set(range(center_start, center_start + num_center)))
    if num_random_lines > 0 and len(available) > 0:
        chosen = rng.choice(available, size=min(num_random_lines, len(available)), replace=False)
        for idx in chosen:
            mask[idx, :] = 1.0

    return mask
