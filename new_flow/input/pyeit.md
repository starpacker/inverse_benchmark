```python
# coding: utf-8
""" 
Self-contained demo on JAC 3D 
Extracts core logic (FEM, JAC, Protocol) from pyEIT for educational purposes.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as la
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Callable

# Keep mesh generation from pyeit as it is a complex geometric utility
import pyeit.mesh as mesh
from pyeit.mesh.shape import ball
from pyeit.mesh.wrapper import PyEITAnomaly_Ball
from pyeit.mesh import PyEITMesh

# ==============================================================================
# 1. Protocol Section (Extracted from pyeit.eit.protocol)
# ==============================================================================

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
        # build [[m, n, idx]_i] array
        m = np.arange(n_el) % n_el
        n = (m + step) % n_el
        idx = exc_id * np.ones(n_el)
        meas_pattern = np.vstack([n, m, idx]).T

        # Filter measurements on current carrying electrodes
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

# ==============================================================================
# 2. Forward Model Section (Extracted from pyeit.eit.fem)
# ==============================================================================

def det2x2(s1: np.ndarray, s2: np.ndarray):
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]

def _k_tetrahedron(xy: np.ndarray):
    """Calculate local stiffness matrix for tetrahedron"""
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]
    # volume of the tetrahedron
    vt = 1.0 / 6 * la.det(s[[0, 1, 2]])
    # calculate area (vector) of triangle faces
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])
    # local stiffness matrix
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

    # set reference nodes
    if 0 <= ref < n_pts:
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))

def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray):
    """Calculate voltage differences based on measurement pattern"""
    idx = meas_pattern[:, 2]
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]

class EITForward:
    """EIT Forward Solver"""
    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol) -> None:
        self.mesh = mesh
        self.protocol = protocol
        # Pre-calculate local stiffness matrices (depends only on geometry)
        self.se = calculate_ke(self.mesh.node, self.mesh.element)
        self.kg = None # Global stiffness matrix

    def assemble_pde(self, perm: Optional[np.ndarray] = None) -> None:
        if perm is None:
            perm = self.mesh.perm_array
        # Ensure perm is valid array
        if not isinstance(perm, np.ndarray):
             perm = perm * np.ones(self.mesh.n_elems)
             
        self.kg = assemble(
            self.se,
            self.mesh.element,
            perm,
            self.mesh.n_nodes,
            ref=self.mesh.ref_node,
        )

    def solve_vectorized(self, ex_mat: np.ndarray) -> np.ndarray:
        """Solve FEM for multiple excitations"""
        # Natural boundary conditions
        b = np.zeros((ex_mat.shape[0], self.mesh.n_nodes))
        b[np.arange(b.shape[0])[:, None], self.mesh.el_pos[ex_mat]] = [1, -1]
        
        result = np.empty((ex_mat.shape[0], self.kg.shape[0]))
        # Solve for each excitation
        for i in range(result.shape[0]):
            result[i] = sparse.linalg.spsolve(self.kg, b[i])
        return result

    def solve_eit(self, perm: Optional[np.ndarray] = None):
        """Simulate boundary voltage measurements"""
        self.assemble_pde(perm)
        f = self.solve_vectorized(self.protocol.ex_mat)
        v = subtract_row_vectorized(f[:, self.mesh.el_pos], self.protocol.meas_mat)
        return v.reshape(-1)

    def compute_jac(self, perm: Optional[np.ndarray] = None):
        """Compute Jacobian matrix"""
        self.assemble_pde(perm)
        # Calculate node potentials f
        f = self.solve_vectorized(self.protocol.ex_mat)
        
        # Calculate r = inv(K) restricted to electrodes (dense, slow but simple)
        # For small 3D meshes this is acceptable. For large ones, adjoint method is better.
        # Here we use the direct inverse method from pyeit source for consistency
        r_mat = la.inv(self.kg.toarray())[self.mesh.el_pos]
        r_el = np.full((self.protocol.ex_mat.shape[0],) + r_mat.shape, r_mat)
        
        # Build measurements and node resistance
        v = subtract_row_vectorized(f[:, self.mesh.el_pos], self.protocol.meas_mat)
        ri = subtract_row_vectorized(r_el, self.protocol.meas_mat)
        v0 = v.reshape(-1)

        # Build Jacobian element-wise
        jac = np.zeros((self.protocol.n_meas, self.mesh.n_elems))
        indices = self.protocol.meas_mat[:, 2]
        f_n = f[indices]
        
        for e, ijk in enumerate(self.mesh.element):
            jac[:, e] = np.sum(np.dot(ri[:, ijk], self.se[e]) * f_n[:, ijk], axis=1)

        return jac, v0

# ==============================================================================
# 3. Inverse Solver Section (Extracted from pyeit.eit.jac)
# ==============================================================================

class JAC:
    """Sensitivity-based EIT imaging (JAC)"""
    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol):
        self.mesh = mesh
        self.protocol = protocol
        self.fwd = EITForward(mesh, protocol)
        self.H = None

    def setup(self, p: float = 0.20, lamb: float = 0.001, method: str = "kotre"):
        """Setup JAC solver"""
        # 1. Compute Jacobian on homogeneous background
        perm = self.mesh.perm
        print("Computing Jacobian...")
        self.J, self.v0 = self.fwd.compute_jac(perm=perm)
        
        # 2. Compute Inverse Matrix H
        print("Computing Inverse Matrix H...")
        self.H = self._compute_h(self.J, p, lamb, method)

    def _compute_h(self, jac: np.ndarray, p: float, lamb: float, method: str):
        j_w_j = np.dot(jac.transpose(), jac)
        
        # Regularization matrix R
        if method == "kotre":
            r_mat = np.diag(np.diag(j_w_j) ** p)
        elif method == "lm":
            r_mat = np.diag(np.diag(j_w_j))
        else: # dgn
            r_mat = np.eye(jac.shape[1])

        # H = (J.T*J + lamb*R)^(-1) * J.T
        return np.dot(la.inv(j_w_j + lamb * r_mat), jac.transpose())

    def solve(self, v1: np.ndarray, v0: np.ndarray, normalize: bool = False):
        """Solve for conductivity changes"""
        # Normalized difference
        if normalize:
            dv = np.log(np.abs(v1) / np.abs(v0)) * np.sign(v0.real)
        else:
            dv = (v1 - v0)
            
        # ds = -H * dv
        return -np.dot(self.H, dv.transpose())

# ==============================================================================
# 4. Utilities (Visualization)
# ==============================================================================

def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray):
    """Interpolate element values to node values for visualization"""
    # Simple averaging: average values of elements sharing a node
    # Note: A more accurate way uses element volumes as weights
    n_nodes = pts.shape[0]
    node_val = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    
    for i, el in enumerate(sim):
        node_val[el] += sim_values[i]
        count[el] += 1
        
    return node_val / np.maximum(count, 1)

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # 1. Mesh Generation (using pyeit library for complex geometry)
    print("Generating Mesh...")
    bbox = [[-1, -1, -1], [1, 1, 1]]
    n_el = 16
    mesh_obj = mesh.create(n_el, h0=0.2, bbox=bbox, fd=ball)
    
    pts = mesh_obj.node
    tri = mesh_obj.element
    print(f"Mesh status: {mesh_obj.n_nodes} nodes, {mesh_obj.n_elems} elements")

    # 2. Protocol Setup
    print("Setting up Protocol...")
    protocol_obj = create_protocol(n_el, dist_exc=7, step_meas=1)

    # 3. Forward Simulation
    print("Running Forward Simulation...")
    fwd = EITForward(mesh_obj, protocol_obj)
    
    # Calculate baseline (homogeneous)
    v0 = fwd.solve_eit()
    
    # Create Anomaly
    print("Adding Anomaly...")
    anomaly = PyEITAnomaly_Ball(center=[0.4, 0.4, 0.0], r=0.3, perm=100.0)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    
    # Calculate perturbed data
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    # 4. Inverse Solving
    print("Running Inverse Solver (JAC)...")
    eit = JAC(mesh_obj, protocol_obj)
    eit.setup(p=0.50, lamb=1e-3, method="kotre")
    
    ds = eit.solve(v1, v0, normalize=False)
    
    # 5. Visualization
    print("Visualizing...")
    # Map element values to nodes for smooth plotting
    node_ds = sim2pts(pts, tri, np.real(ds))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    im = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=node_ds, cmap='viridis', s=50, alpha=0.8)
    fig.colorbar(im, ax=ax, label='Conductivity Change')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D EIT Reconstruction (Self-Contained)')
    
    plt.savefig("3D_eit.png")
    print("Done! Saved to 3D_eit.png")
```
