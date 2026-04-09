import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_cc(ref, est):
    """Pearson correlation coefficient."""
    return float(np.corrcoef(ref, est)[0, 1])
