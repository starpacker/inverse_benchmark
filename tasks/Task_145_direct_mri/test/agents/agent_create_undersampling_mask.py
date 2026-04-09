import matplotlib

matplotlib.use('Agg')

import numpy as np

def create_undersampling_mask(N, acceleration=4, acs_lines=16, seed=42):
    """Create a random Cartesian undersampling mask (row-based)."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((N, N), dtype=np.float64)

    center = N // 2
    acs_start = center - acs_lines // 2
    acs_end = center + acs_lines // 2
    mask[acs_start:acs_end, :] = 1.0

    total_lines_needed = N // acceleration
    acs_count = acs_end - acs_start
    remaining_needed = max(0, total_lines_needed - acs_count)

    available = list(set(range(N)) - set(range(acs_start, acs_end)))
    chosen = rng.choice(available, size=min(remaining_needed, len(available)), replace=False)
    for idx in chosen:
        mask[idx, :] = 1.0

    return mask
