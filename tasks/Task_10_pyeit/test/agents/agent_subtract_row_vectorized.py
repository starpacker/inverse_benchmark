import numpy as np


# --- Extracted Dependencies ---

def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """Calculate voltage differences based on measurement pattern"""
    idx = meas_pattern[:, 2]
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]
