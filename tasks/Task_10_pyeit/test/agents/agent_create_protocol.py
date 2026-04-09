import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# --- Extracted Dependencies ---

class PyEITProtocol:
    """EIT Protocol object"""
    ex_mat: np.ndarray
    meas_mat: np.ndarray
    keep_ba: np.ndarray

    @property
    def n_meas(self) -> int:
        return self.meas_mat.shape[0]

def build_exc_pattern_std(n_el: int = 16, dist: int = 1) -> np.ndarray:
    """Generate scan matrix, `ex_mat` (adjacent mode)"""
    return np.array([[i, np.mod(i + dist, n_el)] for i in range(n_el)])

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

def create_protocol(n_el: int = 16, dist_exc: int = 1, step_meas: int = 1) -> PyEITProtocol:
    """Create a standard EIT protocol"""
    ex_mat = build_exc_pattern_std(n_el, dist_exc)
    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)
