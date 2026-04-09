import numpy as np
import scipy.linalg as la
from scipy import sparse
from dataclasses import dataclass
from typing import Optional, Tuple
import pyeit.mesh as mesh
from pyeit.mesh.shape import ball
from pyeit.mesh.wrapper import PyEITAnomaly_Ball
from pyeit.mesh import PyEITMesh
import matplotlib.pyplot as plt

@dataclass
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

def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """Calculate voltage differences based on measurement pattern"""
    idx = meas_pattern[:, 2]
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]

def load_and_preprocess_data(
    n_el: int,
    h0: float,
    bbox: list,
    dist_exc: int,
    step_meas: int,
    anomaly_center: list,
    anomaly_r: float,
    anomaly_perm: float,
    background_perm: float
) -> dict:
    """
    Load and preprocess data for EIT imaging.
    """
    print("Generating Mesh...")
    mesh_obj = mesh.create(n_el, h0=h0, bbox=bbox, fd=ball)
    
    # FIX: Use dot notation for PyEITMesh object, NOT dictionary access
    pts = mesh_obj.node
    tri = mesh_obj.element
    print(f"Mesh status: {mesh_obj.n_nodes} nodes, {mesh_obj.n_elems} elements")

    print("Setting up Protocol...")
    protocol_obj = create_protocol(n_el, dist_exc=dist_exc, step_meas=step_meas)

    print("Pre-calculating local stiffness matrices...")
    se = calculate_ke(pts, tri)

    print("Computing baseline measurements (homogeneous)...")
    perm_baseline = mesh_obj.perm_array
    if not isinstance(perm_baseline, np.ndarray):
        perm_baseline = mesh_obj.perm * np.ones(mesh_obj.n_elems)
    
    kg_baseline = assemble(se, tri, perm_baseline, mesh_obj.n_nodes, ref=mesh_obj.ref_node)
    
    # FIX: Correct indexing for b_baseline
    b_baseline = np.zeros((protocol_obj.ex_mat.shape[0], mesh_obj.n_nodes))
    b_baseline[np.arange(b_baseline.shape[0])[:, None], mesh_obj.el_pos[protocol_obj.ex_mat]] = [1, -1]
    
    f_baseline = np.empty((protocol_obj.ex_mat.shape[0], kg_baseline.shape[0]))
    for i in range(f_baseline.shape[0]):
        f_baseline[i] = sparse.linalg.spsolve(kg_baseline, b_baseline[i])
    
    v0 = subtract_row_vectorized(f_baseline[:, mesh_obj.el_pos], protocol_obj.meas_mat).reshape(-1)

    print("Adding Anomaly and computing perturbed measurements...")
    anomaly = PyEITAnomaly_Ball(center=anomaly_center, r=anomaly_r, perm=anomaly_perm)
    
    # FIX: mesh.set_perm returns a new mesh object in the reference implementation
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)
    
    perm_anomaly = mesh_new.perm_array
    if not isinstance(perm_anomaly, np.ndarray):
        perm_anomaly = mesh_new.perm * np.ones(mesh_new.n_elems)
    
    kg_anomaly = assemble(se, tri, perm_anomaly, mesh_obj.n_nodes, ref=mesh_obj.ref_node)
    
    f_anomaly = np.empty((protocol_obj.ex_mat.shape[0], kg_anomaly.shape[0]))
    for i in range(f_anomaly.shape[0]):
        f_anomaly[i] = sparse.linalg.spsolve(kg_anomaly, b_baseline[i])
    
    v1 = subtract_row_vectorized(f_anomaly[:, mesh_obj.el_pos], protocol_obj.meas_mat).reshape(-1)

    return {
        'mesh': mesh_obj,
        'protocol': protocol_obj,
        'pts': pts,
        'tri': tri,
        'se': se,
        'v0': v0,
        'v1': v1,
        'perm_baseline': perm_baseline,
        'f_baseline': f_baseline,
        'kg_baseline': kg_baseline
    }