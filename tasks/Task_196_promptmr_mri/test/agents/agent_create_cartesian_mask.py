import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_cartesian_mask(N, acceleration, acs_fraction=0.08, seed=42):
    """
    Create a 1D Cartesian undersampling mask (same for all columns).
    Keeps center ACS lines and randomly selects remaining lines.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros(N, dtype=bool)

    acs_lines = int(N * acs_fraction)
    center = N // 2
    acs_start = center - acs_lines // 2
    acs_end = acs_start + acs_lines
    mask[acs_start:acs_end] = True

    total_lines = N // acceleration
    remaining = max(0, total_lines - acs_lines)

    non_acs_indices = np.where(~mask)[0]
    if remaining > 0 and len(non_acs_indices) > 0:
        chosen = rng.choice(non_acs_indices, size=min(remaining, len(non_acs_indices)), replace=False)
        mask[chosen] = True

    mask_2d = np.zeros((N, N), dtype=bool)
    for i in range(N):
        if mask[i]:
            mask_2d[i, :] = True

    return mask_2d
