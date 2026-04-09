import numpy as np


# --- Extracted Dependencies ---

def build_exc_pattern_std(n_el: int = 16, dist: int = 1) -> np.ndarray:
    """Generate scan matrix, `ex_mat` (adjacent mode)"""
    return np.array([[i, np.mod(i + dist, n_el)] for i in range(n_el)])
