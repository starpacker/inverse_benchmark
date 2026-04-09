import matplotlib

matplotlib.use('Agg')

import numpy as np

def compute_correlation(x, y):
    """Pearson correlation coefficient."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
