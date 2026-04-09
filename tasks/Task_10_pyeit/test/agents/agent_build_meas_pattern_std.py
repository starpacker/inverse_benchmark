import numpy as np
from typing import Optional, Tuple


# --- Extracted Dependencies ---

def build_meas_pattern_std(
    ex_mat: np.ndarray,
    n_el: int = 16,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the measurement pattern"""
    diff_op, keep_ba = [], []
    for exc_id, exc_line in enumerate(ex_mat):
        a, b = exc_line[0], exc_line[1]
        m = np.arange(n_el) % n_el
        n = (m + step) % n_el
        idx = exc_id * np.ones(n_el)
        meas_pattern = np.vstack([n, m, idx]).T

        diff_keep = np.logical_and.reduce((m != a, m != b, n != a, n != b))
        keep_ba.append(diff_keep)
        meas_pattern = meas_pattern[diff_keep]
        diff_op.append(meas_pattern.astype(int))

    return np.vstack(diff_op), np.array(keep_ba).ravel()
