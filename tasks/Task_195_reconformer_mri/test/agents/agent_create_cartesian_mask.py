import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_cartesian_mask(N, acceleration=4, acs_fraction=0.08, seed=42):
    """Create 1D Cartesian undersampling mask with ACS lines."""
    rng = np.random.RandomState(seed)
    mask_1d = np.zeros(N, dtype=np.float64)
    
    acs_n = int(N * acs_fraction)
    c0 = N // 2 - acs_n // 2
    mask_1d[c0:c0 + acs_n] = 1.0
    
    target = N // acceleration
    needed = target - acs_n
    available = np.setdiff1d(np.arange(N), np.arange(c0, c0 + acs_n))
    if needed > 0:
        chosen = rng.choice(available, min(needed, len(available)), replace=False)
        mask_1d[chosen] = 1.0
    
    mask_2d = np.tile(mask_1d[:, None], (1, N))
    rate = mask_1d.sum() / N
    print(f"  Undersampling mask: {int(mask_1d.sum())}/{N} lines "
          f"({rate*100:.1f}%), ~{1/rate:.1f}x acceleration")
    return mask_2d
