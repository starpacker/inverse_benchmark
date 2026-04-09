"""
Self-contained demo on JAC 3D
Refactored into 4 functional components for EIT imaging/inversion.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as la
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Optional, Tuple

import pyeit.mesh as mesh
from pyeit.mesh.shape import ball
from pyeit.mesh.wrapper import PyEITAnomaly_Ball
from pyeit.mesh import PyEITMesh


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


def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    """Interpolate element values to node values for visualization"""
    n_nodes = pts.shape[0]
    node_val = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    
    for i, el in enumerate(sim):
        node_val[el] += sim_values[i]
        count[el] += 1
        
    return node_val / np.maximum(count, 1)


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
    
    Creates mesh, protocol, and generates baseline and perturbed measurements.
    
    Parameters
    ----------
    n_el : int
        Number of electrodes
    h0 : float
        Mesh element size parameter
    bbox : list
        Bounding box for mesh generation
    dist_exc : int
        Distance between excitation electrodes
    step_meas : int
        Step for measurement pattern
    anomaly_center : list
        Center of the anomaly [x, y, z]
    anomaly_r : float
        Radius of the anomaly
    anomaly_perm : float
        Permittivity of the anomaly
    background_perm : float
        Background permittivity
    
    Returns
    -------
    dict
        Dictionary containing mesh, protocol, measurements, and precomputed matrices
    """
    print("Generating Mesh...")
    mesh_obj = mesh.create(n_el, h0=h0, bbox=bbox, fd=ball)
    
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
    
    b_baseline = np.zeros((protocol_obj.ex_mat.shape[0], mesh_obj.n_nodes))
    b_baseline[np.arange(b_baseline.shape[0])[:, None], mesh_obj.el_pos[protocol_obj.ex_mat]] = [1, -1]
    
    f_baseline = np.empty((protocol_obj.ex_mat.shape[0], kg_baseline.shape[0]))
    for i in range(f_baseline.shape[0]):
        f_baseline[i] = sparse.linalg.spsolve(kg_baseline, b_baseline[i])
    
    v0 = subtract_row_vectorized(f_baseline[:, mesh_obj.el_pos], protocol_obj.meas_mat).reshape(-1)

    print("Adding Anomaly and computing perturbed measurements...")
    anomaly = PyEITAnomaly_Ball(center=anomaly_center, r=anomaly_r, perm=anomaly_perm)
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


def forward_operator(
    x: np.ndarray,
    mesh_obj: PyEITMesh,
    protocol_obj: PyEITProtocol,
    se: np.ndarray
) -> np.ndarray:
    """
    Forward operator: compute boundary voltages from conductivity distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Conductivity distribution (element-wise permittivity values)
    mesh_obj : PyEITMesh
        Mesh object
    protocol_obj : PyEITProtocol
        Protocol object
    se : np.ndarray
        Pre-computed local stiffness matrices
    
    Returns
    -------
    np.ndarray
        Predicted boundary voltage measurements
    """
    pts = mesh_obj.node
    tri = mesh_obj.element
    n_nodes = mesh_obj.n_nodes
    ref_node = mesh_obj.ref_node
    el_pos = mesh_obj.el_pos
    ex_mat = protocol_obj.ex_mat
    meas_mat = protocol_obj.meas_mat

    perm = x
    if not isinstance(perm, np.ndarray):
        perm = perm * np.ones(mesh_obj.n_elems)

    kg = assemble(se, tri, perm, n_nodes, ref=ref_node)

    b = np.zeros((ex_mat.shape[0], n_nodes))
    b[np.arange(b.shape[0])[:, None], el_pos[ex_mat]] = [1, -1]

    f = np.empty((ex_mat.shape[0], kg.shape[0]))
    for i in range(f.shape[0]):
        f[i] = sparse.linalg.spsolve(kg, b[i])

    v = subtract_row_vectorized(f[:, el_pos], meas_mat)
    return v.reshape(-1)


def run_inversion(
    v1: np.ndarray,
    v0: np.ndarray,
    mesh_obj: PyEITMesh,
    protocol_obj: PyEITProtocol,
    se: np.ndarray,
    perm_baseline: np.ndarray,
    p: float = 0.20,
    lamb: float = 0.001,
    method: str = "kotre",
    normalize: bool = False
) -> np.ndarray:
    """
    Run the JAC inversion algorithm to reconstruct conductivity changes.
    
    Parameters
    ----------
    v1 : np.ndarray
        Perturbed boundary voltage measurements
    v0 : np.ndarray
        Baseline boundary voltage measurements
    mesh_obj : PyEITMesh
        Mesh object
    protocol_obj : PyEITProtocol
        Protocol object
    se : np.ndarray
        Pre-computed local stiffness matrices
    perm_baseline : np.ndarray
        Baseline permittivity distribution
    p : float
        Regularization parameter for Kotre method
    lamb : float
        Regularization parameter (lambda)
    method : str
        Regularization method ('kotre', 'lm', or 'dgn')
    normalize : bool
        Whether to use normalized difference
    
    Returns
    -------
    np.ndarray
        Reconstructed conductivity change (element-wise)
    """
    print("Computing Jacobian...")
    pts = mesh_obj.node
    tri = mesh_obj.element
    n_nodes = mesh_obj.n_nodes
    n_elems = mesh_obj.n_elems
    ref_node = mesh_obj.ref_node
    el_pos = mesh_obj.el_pos
    ex_mat = protocol_obj.ex_mat
    meas_mat = protocol_obj.meas_mat
    n_meas = protocol_obj.n_meas

    perm = perm_baseline
    if not isinstance(perm, np.ndarray):
        perm = perm * np.ones(n_elems)

    kg = assemble(se, tri, perm, n_nodes, ref=ref_node)

    b = np.zeros((ex_mat.shape[0], n_nodes))
    b[np.arange(b.shape[0])[:, None], el_pos[ex_mat]] = [1, -1]

    f = np.empty((ex_mat.shape[0], kg.shape[0]))
    for i in range(f.shape[0]):
        f[i] = sparse.linalg.spsolve(kg, b[i])

    r_mat = la.inv(kg.toarray())[el_pos]
    r_el = np.full((ex_mat.shape[0],) + r_mat.shape, r_mat)

    ri = subtract_row_vectorized(r_el, meas_mat)

    jac = np.zeros((n_meas, n_elems))
    indices = meas_mat[:, 2]
    f_n = f[indices]

    for e, ijk in enumerate(tri):
        jac[:, e] = np.sum(np.dot(ri[:, ijk], se[e]) * f_n[:, ijk], axis=1)

    print("Computing Inverse Matrix H...")
    j_w_j = np.dot(jac.transpose(), jac)

    if method == "kotre":
        r_mat_reg = np.diag(np.diag(j_w_j) ** p)
    elif method == "lm":
        r_mat_reg = np.diag(np.diag(j_w_j))
    else:
        r_mat_reg = np.eye(jac.shape[1])

    H = np.dot(la.inv(j_w_j + lamb * r_mat_reg), jac.transpose())

    print("Solving inverse problem...")
    if normalize:
        dv = np.log(np.abs(v1) / np.abs(v0)) * np.sign(v0.real)
    else:
        dv = (v1 - v0)

    ds = -np.dot(H, dv.transpose())

    return ds


def evaluate_results(
    ds: np.ndarray,
    pts: np.ndarray,
    tri: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    output_filename: str = "3D_eit.png"
) -> dict:
    """
    Evaluate and visualize the reconstruction results.
    
    Parameters
    ----------
    ds : np.ndarray
        Reconstructed conductivity change (element-wise)
    pts : np.ndarray
        Node coordinates
    tri : np.ndarray
        Element connectivity
    v0 : np.ndarray
        Baseline measurements
    v1 : np.ndarray
        Perturbed measurements
    output_filename : str
        Output filename for the visualization
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    print("Evaluating results...")

    node_ds = sim2pts(pts, tri, np.real(ds))

    ds_real = np.real(ds)
    ds_min = np.min(ds_real)
    ds_max = np.max(ds_real)
    ds_mean = np.mean(ds_real)
    ds_std = np.std(ds_real)

    dv = v1 - v0
    data_residual = np.linalg.norm(dv)
    relative_change = np.linalg.norm(dv) / np.linalg.norm(v0) if np.linalg.norm(v0) > 0 else 0.0

    max_change_element = np.argmax(np.abs(ds_real))
    max_change_value = ds_real[max_change_element]

    print(f"Reconstruction statistics:")
    print(f"  Min conductivity change: {ds_min:.6f}")
    print(f"  Max conductivity change: {ds_max:.6f}")
    print(f"  Mean conductivity change: {ds_mean:.6f}")
    print(f"  Std conductivity change: {ds_std:.6f}")
    print(f"  Data residual norm: {data_residual:.6f}")
    print(f"  Relative voltage change: {relative_change:.6f}")
    print(f"  Element with max change: {max_change_element} (value: {max_change_value:.6f})")

    print("Visualizing...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    im = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=node_ds, cmap='viridis', s=50, alpha=0.8)
    fig.colorbar(im, ax=ax, label='Conductivity Change')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D EIT Reconstruction (Refactored)')

    plt.savefig(output_filename)
    print(f"Saved visualization to {output_filename}")

    metrics = {
        'ds_min': ds_min,
        'ds_max': ds_max,
        'ds_mean': ds_mean,
        'ds_std': ds_std,
        'data_residual': data_residual,
        'relative_change': relative_change,
        'max_change_element': max_change_element,
        'max_change_value': max_change_value,
        'node_ds': node_ds
    }

    return metrics


if __name__ == "__main__":
    n_el = 16
    h0 = 0.2
    bbox = [[-1, -1, -1], [1, 1, 1]]
    dist_exc = 7
    step_meas = 1
    anomaly_center = [0.4, 0.4, 0.0]
    anomaly_r = 0.3
    anomaly_perm = 100.0
    background_perm = 1.0

    p = 0.50
    lamb = 1e-3
    method = "kotre"
    normalize = False
    output_filename = "3D_eit.png"

    data = load_and_preprocess_data(
        n_el=n_el,
        h0=h0,
        bbox=bbox,
        dist_exc=dist_exc,
        step_meas=step_meas,
        anomaly_center=anomaly_center,
        anomaly_r=anomaly_r,
        anomaly_perm=anomaly_perm,
        background_perm=background_perm
    )

    mesh_obj = data['mesh']
    protocol_obj = data['protocol']
    pts = data['pts']
    tri = data['tri']
    se = data['se']
    v0 = data['v0']
    v1 = data['v1']
    perm_baseline = data['perm_baseline']

    print("Testing forward operator...")
    v_pred = forward_operator(
        x=perm_baseline,
        mesh_obj=mesh_obj,
        protocol_obj=protocol_obj,
        se=se
    )
    print(f"Forward operator output shape: {v_pred.shape}")
    print(f"Forward operator consistency check (should be ~0): {np.linalg.norm(v_pred - v0):.6e}")

    print("Running Inverse Solver (JAC)...")
    ds = run_inversion(
        v1=v1,
        v0=v0,
        mesh_obj=mesh_obj,
        protocol_obj=protocol_obj,
        se=se,
        perm_baseline=perm_baseline,
        p=p,
        lamb=lamb,
        method=method,
        normalize=normalize
    )

    metrics = evaluate_results(
        ds=ds,
        pts=pts,
        tri=tri,
        v0=v0,
        v1=v1,
        output_filename=output_filename
    )

    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")