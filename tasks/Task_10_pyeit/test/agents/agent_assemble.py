import numpy as np
from scipy import sparse


# --- Extracted Dependencies ---

def assemble(ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0):
    """Assemble the global stiffness matrix"""
    n_tri, n_vertices = tri.shape
    row = np.repeat(tri, n_vertices).ravel()
    col = np.repeat(tri, n_vertices, axis=0).ravel()
    data = np.array([ke[i] * perm[i] for i in range(n_tri)]).ravel()

    if 0 <= ref < n_pts:
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))
