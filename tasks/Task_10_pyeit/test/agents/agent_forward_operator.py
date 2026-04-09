import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from dataclasses import dataclass
from pyeit.mesh import PyEITMesh

@dataclass
class PyEITProtocol:
    """
    EIT Protocol object containing stimulation and measurement patterns.
    
    Attributes:
        ex_mat (np.ndarray): Excitation matrix (N_excitations x 2). 
                             Each row contains [source_electrode, sink_electrode].
        meas_mat (np.ndarray): Measurement matrix (N_measurements x 3).
                               Each row contains [positive_electrode, negative_electrode, excitation_index].
        keep_ba (np.ndarray): Boolean array indicating valid measurements.
    """
    ex_mat: np.ndarray
    meas_mat: np.ndarray
    keep_ba: np.ndarray

    @property
    def n_meas(self) -> int:
        """Returns the total number of measurements defined."""
        return self.meas_mat.shape[0]

def assemble(ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0):
    """
    Assemble the global stiffness matrix K.

    Parameters:
        ke (np.ndarray): Local stiffness matrices for master elements.
        tri (np.ndarray): Mesh topology (N_triangles x 3 vertices).
        perm (np.ndarray): Conductivity/Permittivity of each element (N_triangles).
        n_pts (int): Total number of nodes in the mesh.
        ref (int): Index of the reference node to ground (default 0).

    Returns:
        sparse.csr_matrix: The assembled global stiffness matrix (n_pts x n_pts).
    """
    n_tri, n_vertices = tri.shape
    
    # 1. Map local indices to global indices
    row = np.repeat(tri, n_vertices).ravel()
    col = np.repeat(tri, n_vertices, axis=0).ravel()
    
    # 2. Scale local stiffness by element conductivity
    data = np.array([ke[i] * perm[i] for i in range(n_tri)]).ravel()

    # 3. Apply Reference Node Constraint (Grounding)
    if 0 <= ref < n_pts:
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    # 4. Create Compressed Sparse Row matrix
    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))

def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """
    Calculate voltage differences based on measurement pattern.

    Parameters:
        v (np.ndarray): Calculated potentials at electrode nodes.
                        Shape: (N_excitations, N_electrodes)
        meas_pattern (np.ndarray): Measurement pattern matrix.
                                   Columns: [P_electrode, N_electrode, Excitation_Index]

    Returns:
        np.ndarray: Vector of voltage differences.
    """
    idx = meas_pattern[:, 2]
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]

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
        Conductivity distribution (element-wise permittivity values).
    mesh_obj : PyEITMesh
        Mesh object containing geometry and electrode positions.
    protocol_obj : PyEITProtocol
        Protocol object defining excitation and measurement patterns.
    se : np.ndarray
        Pre-computed local stiffness matrices (shape: N_elems x 3 x 3).
    
    Returns
    -------
    np.ndarray
        Predicted boundary voltage measurements (1D array).
    """
    # --- 1. Setup Geometry ---
    pts = mesh_obj.node
    tri = mesh_obj.element
    n_nodes = mesh_obj.n_nodes
    ref_node = mesh_obj.ref_node
    el_pos = mesh_obj.el_pos
    ex_mat = protocol_obj.ex_mat
    meas_mat = protocol_obj.meas_mat

    # --- 2. Prepare Conductivity ---
    perm = x
    if not isinstance(perm, np.ndarray):
        perm = perm * np.ones(mesh_obj.n_elems)

    # --- 3. Assemble Global Stiffness Matrix ---
    kg = assemble(se, tri, perm, n_nodes, ref=ref_node)

    # --- 4. Construct Current Source Vectors (RHS) ---
    b = np.zeros((ex_mat.shape[0], n_nodes))
    b[np.arange(b.shape[0])[:, None], el_pos[ex_mat]] = [1, -1]

    # --- 5. Solve Linear System (Ku = b) ---
    f = np.empty((ex_mat.shape[0], kg.shape[0]))
    for i in range(f.shape[0]):
        f[i] = sparse.linalg.spsolve(kg, b[i])

    # --- 6. Extract Measurements ---
    v = subtract_row_vectorized(f[:, el_pos], meas_mat)
    
    return v.reshape(-1)