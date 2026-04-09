import numpy as np

import matplotlib

matplotlib.use('Agg')

def to_db(source_map, dynamic_range=30.0):
    """Convert to dB with dynamic range."""
    mx = np.max(source_map)
    if mx <= 0:
        return np.full_like(source_map, -dynamic_range)
    n = np.maximum(source_map / mx, 10 ** (-dynamic_range / 10))
    return 10.0 * np.log10(n)
