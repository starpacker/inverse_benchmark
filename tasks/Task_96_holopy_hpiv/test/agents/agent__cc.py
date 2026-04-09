import numpy as np

import matplotlib

matplotlib.use("Agg")

def _cc(a, b):
    """Pearson Correlation Coefficient."""
    af, bf = a.ravel(), b.ravel()
    if np.std(af) > 1e-15 and np.std(bf) > 1e-15:
        return float(np.corrcoef(af, bf)[0, 1])
    return 0.0
