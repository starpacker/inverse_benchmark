import numpy as np


# --- Extracted Dependencies ---

def make_laplace_kernel(rows, columns, dtype='float32'):
    """Generate eigenvalues of diagonalized Laplacian operator"""
    xi_y = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows)).reshape((-1, 1))
    xi_x = (2 - 2 * np.cos(np.pi * np.arange(columns) / columns)).reshape((1, -1))
    eigvals = xi_y + xi_x

    with np.errstate(divide="ignore"):
        K = np.nan_to_num(1 / eigvals, posinf=0, neginf=0)
    return K.astype(dtype)
