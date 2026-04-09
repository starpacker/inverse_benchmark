import numpy as np
import scipy.linalg as la


# --- Extracted Dependencies ---

def _k_tetrahedron(xy: np.ndarray) -> np.ndarray:
    """Calculate local stiffness matrix for tetrahedron"""
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]
    vt = 1.0 / 6 * la.det(s[[0, 1, 2]])
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])
    return np.dot(a, a.transpose()) / (36.0 * vt)

def calculate_ke(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Calculate local stiffness matrix on all elements."""
    n_tri, n_vertices = tri.shape
    if n_vertices != 4:
        raise TypeError("This demo supports 3D tetrahedrons (4 vertices) only.")
    
    ke_array = np.zeros((n_tri, n_vertices, n_vertices))
    for ei in range(n_tri):
        no = tri[ei, :]
        xy = pts[no]
        ke_array[ei] = _k_tetrahedron(xy)
    return ke_array
