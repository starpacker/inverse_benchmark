import numpy as np


# --- Extracted Dependencies ---

def est_wrapped_gradient(arr, Dx, Dy, dtype=np.float32):
    """Estimate the wrapped gradient of `arr` using differential operators `Dx, Dy`"""
    rows, columns = arr.shape

    phi_x = (Dx @ arr.ravel()).reshape((rows, columns))
    phi_y = (Dy @ arr.ravel()).reshape((rows, columns))
    # Make wrapped adjustment (eq. (2), (3))
    idxs = np.abs(phi_x) > np.pi
    phi_x[idxs] -= 2 * np.pi * np.sign(phi_x[idxs])
    idxs = np.abs(phi_y) > np.pi
    phi_y[idxs] -= 2 * np.pi * np.sign(phi_y[idxs])
    return phi_x, phi_y
