import numpy as np
import scipy.linalg as la
from scipy import sparse
from dataclasses import dataclass
from pyeit.mesh import PyEITMesh

# --- Data Structures ---

@dataclass
class PyEITProtocol:
    """EIT Protocol object container."""
    ex_mat: np.ndarray    # Excitation patterns
    meas_mat: np.ndarray  # Measurement patterns
    keep_ba: np.ndarray   # Boolean array for valid measurements

    @property
    def n_meas(self) -> int:
        return self.meas_mat.shape[0]

# --- Core Math Functions ---

def assemble(ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0):
    """
    Assemble the global stiffness matrix K from local element matrices.
    
    Args:
        ke: Local stiffness matrices (n_tri, 3, 3)
        tri: Triangle connectivity (n_tri, 3)
        perm: Permittivity per element (n_tri,)
        n_pts: Total number of nodes
        ref: Reference node index to ground (Dirichlet BC)
    """
    n_tri, n_vertices = tri.shape
    # Create row and column indices for the sparse matrix
    row = np.repeat(tri, n_vertices).ravel()
    col = np.repeat(tri, n_vertices, axis=0).ravel()
    # Scale local stiffness by permittivity
    data = np.array([ke[i] * perm[i] for i in range(n_tri)]).ravel()

    # Apply Reference Node Boundary Condition
    if 0 <= ref < n_pts:
        # Find indices involving the reference node
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        
        # Remove existing entries for the reference node
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        
        # Set K[ref, ref] = 1.0
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))

def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """
    Calculate voltage differences V_pos - V_neg based on measurement pattern.
    
    Args:
        v: Potentials (n_excitations, n_nodes) or similar
        meas_pattern: (n_meas, 3) array -> [pos_node, neg_node, excitation_idx]
    """
    # meas_pattern[:, 2] is the excitation index corresponding to the measurement
    idx = meas_pattern[:, 2]
    # Vectorized subtraction
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]

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
    Run the JAC inversion algorithm (Gauss-Newton one-step).
    """
    print("Computing Jacobian...")
    # 1. Unpack Mesh and Protocol
    pts = mesh_obj.node
    tri = mesh_obj.element
    n_nodes = mesh_obj.n_nodes
    n_elems = mesh_obj.n_elems
    ref_node = mesh_obj.ref_node
    el_pos = mesh_obj.el_pos
    ex_mat = protocol_obj.ex_mat
    meas_mat = protocol_obj.meas_mat
    n_meas = protocol_obj.n_meas

    # 2. Prepare Permittivity
    perm = perm_baseline
    if not isinstance(perm, np.ndarray):
        perm = perm * np.ones(n_elems)

    # 3. Forward Solver Setup
    # Assemble global stiffness matrix
    kg = assemble(se, tri, perm, n_nodes, ref=ref_node)

    # Build RHS (Current Injection Vector)
    b = np.zeros((ex_mat.shape[0], n_nodes))
    # Inject +1 at positive electrode, -1 at negative electrode
    b[np.arange(b.shape[0])[:, None], el_pos[ex_mat]] = [1, -1]

    # Solve Forward Problem: K * f = b
    f = np.empty((ex_mat.shape[0], kg.shape[0]))
    for i in range(f.shape[0]):
        f[i] = sparse.linalg.spsolve(kg, b[i])

    # 4. Adjoint Solver Setup (for Jacobian)
    # Invert K (dense) restricted to electrode positions for efficiency
    # Note: For very large meshes, full inversion is costly; this assumes moderate size.
    r_mat = la.inv(kg.toarray())[el_pos]
    r_el = np.full((ex_mat.shape[0],) + r_mat.shape, r_mat)

    # Calculate adjoint fields (measurement patterns)
    ri = subtract_row_vectorized(r_el, meas_mat)

    # 5. Compute Jacobian Matrix
    jac = np.zeros((n_meas, n_elems))
    indices = meas_mat[:, 2] # Excitation index for each measurement
    f_n = f[indices]         # Forward field specific to that measurement

    # Iterate over elements to compute sensitivity
    for e, ijk in enumerate(tri):
        # Sensitivity = dot(Adjoint_Grad, Forward_Grad)
        # Implemented via local stiffness matrix multiplication
        jac[:, e] = np.sum(np.dot(ri[:, ijk], se[e]) * f_n[:, ijk], axis=1)

    # 6. Regularization Setup
    print("Computing Inverse Matrix H...")
    j_w_j = np.dot(jac.transpose(), jac) # Approximate Hessian

    if method == "kotre":
        # Kotre regularization: scales by diagonal sensitivity power
        r_mat_reg = np.diag(np.diag(j_w_j) ** p)
    elif method == "lm":
        # Levenberg-Marquardt: scales by diagonal
        r_mat_reg = np.diag(np.diag(j_w_j))
    else:
        # Tikhonov: Identity matrix
        r_mat_reg = np.eye(jac.shape[1])

    # 7. Compute Reconstruction Matrix H
    # H = (J.T * J + lambda * R)^-1 * J.T
    H = np.dot(la.inv(j_w_j + lamb * r_mat_reg), jac.transpose())

    # 8. Solve for Conductivity Change
    print("Solving inverse problem...")
    if normalize:
        # Log-normalized difference
        dv = np.log(np.abs(v1) / np.abs(v0)) * np.sign(v0.real)
    else:
        # Standard difference
        dv = (v1 - v0)

    # ds = -H * dv
    ds = -np.dot(H, dv.transpose())

    return ds