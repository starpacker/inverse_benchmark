import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import os
import sys
import copy

# --- Helper Functions ---

def boundary_attenuation(n_incidence, n_transmission=1.0):
    """Calculate the boundary attenuation factor A using Fresnel's law (Robin BC)."""
    n = n_incidence / n_transmission
    R0 = ((n - 1.) ** 2) / ((n + 1.) ** 2)
    theta_c = np.arcsin(1.0 / n)
    cos_theta_c = np.cos(theta_c)
    A = (2.0 / (1.0 - R0) - 1.0 + np.abs(cos_theta_c) ** 3) / (1.0 - np.abs(cos_theta_c) ** 2)
    return A

class StndMesh:
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.bndvtx = None
        self.mua = None
        self.kappa = None
        self.ri = None
        self.mus = None
        self.ksi = None 
        self.c = None 
        self.source = {}
        self.meas = {}
        self.link = None
        self.dimension = 2
        self.vol = {}

    def copy_from(self, other):
        self.nodes = copy.deepcopy(other.nodes)
        self.elements = copy.deepcopy(other.elements)
        self.bndvtx = copy.deepcopy(other.bndvtx)
        self.mua = copy.deepcopy(other.mua)
        self.kappa = copy.deepcopy(other.kappa)
        self.ri = copy.deepcopy(other.ri)
        self.mus = copy.deepcopy(other.mus)
        self.ksi = copy.deepcopy(other.ksi)
        self.c = copy.deepcopy(other.c)
        self.source = copy.deepcopy(other.source)
        self.meas = copy.deepcopy(other.meas)
        self.link = copy.deepcopy(other.link)
        self.dimension = other.dimension
        self.vol = copy.deepcopy(other.vol)

def assemble_fem(mesh, omega=0):
    """Assemble the FEM system matrix (K + C + B + i*omega*M)."""
    nnodes = mesh.nodes.shape[0]
    
    # 1. Coordinate arrays
    el_nodes = mesh.elements # (nelems, 3)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    
    x1 = x[el_nodes[:, 0]]; y1 = y[el_nodes[:, 0]]
    x2 = x[el_nodes[:, 1]]; y2 = y[el_nodes[:, 1]]
    x3 = x[el_nodes[:, 2]]; y3 = y[el_nodes[:, 2]]
    
    # 2. Area Calculation (2*Area)
    detJ = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    area = 0.5 * np.abs(detJ)
    
    # 3. Gradients of shape functions
    b1 = y2 - y3; c1 = x3 - x2
    b2 = y3 - y1; c2 = x1 - x3
    b3 = y1 - y2; c3 = x2 - x1
    
    kappa_elem = np.mean(mesh.kappa[el_nodes], axis=1)
    mua_elem = np.mean(mesh.mua[el_nodes], axis=1)
    c_elem = np.mean(mesh.c[el_nodes], axis=1)
    
    # --- Stiffness Matrix K ---
    k_factor = kappa_elem / (4.0 * area)
    B = np.stack([b1, b2, b3], axis=1) 
    C = np.stack([c1, c2, c3], axis=1)
    BB = B[:, :, None] * B[:, None, :]
    CC = C[:, :, None] * C[:, None, :]
    K_local = (BB + CC) * k_factor[:, None, None]
    
    # --- Mass Matrix C and M ---
    mass_pattern = np.ones((3,3)) + np.eye(3) 
    C_factor = (area * mua_elem / 12.0)[:, None, None]
    M_factor = (area / c_elem / 12.0)[:, None, None]
    C_local = C_factor * mass_pattern[None, :, :]
    M_local = M_factor * mass_pattern[None, :, :]
    
    A_local = K_local + C_local + 1j * omega * M_local
    
    # Assembly indices
    rows = np.tile(el_nodes[:, :, None], (1, 1, 3)).flatten()
    cols = np.tile(el_nodes[:, None, :], (1, 3, 1)).flatten()
    data = A_local.flatten()
    
    # --- Boundary Matrix B ---
    edges = np.vstack([
        el_nodes[:, [0, 1]],
        el_nodes[:, [1, 2]],
        el_nodes[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    
    # Only edges where both vertices are on the boundary
    is_bnd_node1 = mesh.bndvtx[edges[:,0]] == 1
    is_bnd_node2 = mesh.bndvtx[edges[:,1]] == 1
    is_bnd_edge_cand = is_bnd_node1 & is_bnd_node2
    cand_edges = edges[is_bnd_edge_cand]

    # Find unique edges that appear exactly once (boundary edges)
    # Note: If an edge is internal but connects two boundary nodes (unlikely in convex hull but possible),
    # it appears twice. Boundary edges appear once.
    # We must filter cand_edges for uniqueness in the global list of edges.
    
    # First, let's get counts of all edges to identify boundary ones globally
    # It's expensive to sort all edges again, but safer.
    # However, 'edges' contains all edges from all elements.
    # Sort the rows of edges so (i,j) is same as (j,i) - already done above.
    
    # unique_all, counts_all = np.unique(edges, axis=0, return_counts=True)
    # bnd_edges_global = unique_all[counts_all == 1]
    
    # But checking bndvtx is also a constraint.
    # Intersection: edges that are globally unique AND connect two boundary nodes.
    # Let's do the unique check on the full list first.
    
    # To optimize: convert edges to structured array or void for unique
    # Here we use standard unique
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    
    # Calculate lengths and ksi
    ex1 = x[boundary_edges[:, 0]]; ey1 = y[boundary_edges[:, 0]]
    ex2 = x[boundary_edges[:, 1]]; ey2 = y[boundary_edges[:, 1]]
    lengths = np.sqrt((ex1-ex2)**2 + (ey1-ey2)**2)
    
    ksi_avg = 0.5 * (mesh.ksi[boundary_edges[:, 0]] + mesh.ksi[boundary_edges[:, 1]])
    
    b_factor = (lengths * ksi_avg / 6.0)[:, None, None]
    b_pattern = np.array([[2, 1], [1, 2]])
    B_local = b_factor * b_pattern[None, :, :]
    
    b_rows = np.tile(boundary_edges[:, :, None], (1, 1, 2)).flatten()
    b_cols = np.tile(boundary_edges[:, None, :], (1, 2, 1)).flatten()
    b_data = B_local.flatten()
    
    full_rows = np.concatenate([rows, b_rows])
    full_cols = np.concatenate([cols, b_cols])
    full_data = np.concatenate([data, b_data])
    
    sys_matrix = sp.coo_matrix((full_data, (full_rows, full_cols)), shape=(nnodes, nnodes))
    return sys_matrix.tocsc()

def get_source_vector(mesh):
    """Construct source vector (RHS)"""
    nnodes = mesh.nodes.shape[0]
    
    # Sanitize inputs: ensure link is integer
    if mesh.link is None:
        raise ValueError("Mesh link not defined")
    
    # Active sources
    active_mask = mesh.link[:, 2] == 1
    active_src_idx = np.unique(mesh.link[active_mask, 0]).astype(int)
    nsources = len(active_src_idx)
    
    src_int_func = mesh.source['int_func'] # (all_sources, 4)
    
    # Handle NaNs in source int_func if any (though typically should be clean)
    if np.isnan(src_int_func).any():
        # Fallback or error. For this context, assume correct indices in col 0.
        # But indices must be integers.
        pass

    Q = sp.lil_matrix((nnodes, nsources), dtype=complex)
    
    for i, src_idx in enumerate(active_src_idx):
        entry = src_int_func[src_idx]
        elem_idx = int(entry[0])
        weights = entry[1:]
        el_nodes = mesh.elements[elem_idx]
        for j, node_idx in enumerate(el_nodes):
            Q[node_idx, i] = weights[j]
            
    return Q.tocsc()


# --- Core Functions ---

def load_and_preprocess_data(mesh_path, anomaly_center=None, anomaly_radius=None, anomaly_factor=1.1, grid_step=2.0):
    """
    1. Loads the mesh.
    2. Creates a second mesh with an anomaly (if specified) to simulate data.
    3. Generates the interpolation matrix for the grid.
    Returns: (mesh_baseline, mesh_anomaly, grid_info)
    """
    if not os.path.exists(mesh_path + '.node'):
        raise FileNotFoundError(f"Mesh files not found at {mesh_path}")

    # Load Baseline Mesh
    mesh = StndMesh()
    base = os.path.splitext(mesh_path)[0]
    
    # Load Nodes
    node_data = np.genfromtxt(base + '.node', delimiter='\t')
    mesh.bndvtx = node_data[:, 0].astype(int)
    mesh.nodes = node_data[:, 1:]
    mesh.dimension = mesh.nodes.shape[1]
    
    # Load Elements
    elem_data = np.genfromtxt(base + '.elem', delimiter='\t', dtype=int)
    mesh.elements = elem_data - 1 
    
    # Load Params
    try:
        param_data = np.genfromtxt(base + '.param', skip_header=1)
    except:
        param_data = np.genfromtxt(base + '.param')
            
    mesh.mua = param_data[:, 0]
    mesh.kappa = param_data[:, 1]
    mesh.ri = param_data[:, 2]
    
    # Derived params
    mesh.mus = (1.0 / mesh.kappa) / 3.0 - mesh.mua
    c0 = 299792458000.0
    mesh.c = c0 / mesh.ri
    
    A_val = boundary_attenuation(mesh.ri, 1.0)
    mesh.ksi = 1.0 / (2 * A_val)
    
    # Load Sources
    if os.path.isfile(base + '.source'):
        with open(base + '.source', 'r') as f:
            header = f.readline()
            skip = 1 if 'fixed' in header else 0
        src_data = np.genfromtxt(base + '.source', skip_header=skip+1) 
        mesh.source['coord'] = src_data[:, 1:3]
        mesh.source['int_func'] = src_data[:, 4:] 
        mesh.source['int_func'][:, 0] -= 1
        
    # Load Detectors
    if os.path.isfile(base + '.meas'):
        meas_data = np.genfromtxt(base + '.meas', skip_header=2)
        mesh.meas['coord'] = meas_data[:, 1:3]
        mesh.meas['int_func'] = meas_data[:, 3:]
        mesh.meas['int_func'][:, 0] -= 1
        
    # Load Link
    if os.path.isfile(base + '.link'):
        link_data = np.genfromtxt(base + '.link', skip_header=1)
        mesh.link = link_data
        mesh.link[:, 0:2] -= 1

    # Create Anomaly Mesh
    mesh_anomaly = StndMesh()
    mesh_anomaly.copy_from(mesh)
    
    if anomaly_center is not None:
        dist = np.linalg.norm(mesh.nodes - np.array(anomaly_center), axis=1)
        idx = np.where(dist < anomaly_radius)[0]
        # Change MUA
        mesh_anomaly.mua[idx] *= anomaly_factor
        # Update derived
        mesh_anomaly.kappa[idx] = 1.0 / (3.0 * (mesh_anomaly.mua[idx] + mesh_anomaly.mus[idx]))
        # c and ksi depend on RI, assumed constant here for anomaly

    # Generate Grid Info
    x_min, x_max = np.min(mesh.nodes[:,0]), np.max(mesh.nodes[:,0])
    y_min, y_max = np.min(mesh.nodes[:,1]), np.max(mesh.nodes[:,1])
    # Slight buffer
    xgrid = np.arange(x_min, x_max, grid_step)
    ygrid = np.arange(y_min, y_max, grid_step)
    
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    tri = spatial.Delaunay(mesh.nodes)
    simplex_indices = tri.find_simplex(grid_points)
    valid = simplex_indices != -1
    valid_simplices = simplex_indices[valid]
    
    b = tri.transform[valid_simplices, :2]
    c = tri.transform[valid_simplices, 2]
    coords = grid_points[valid] - c
    bary = np.einsum('ijk,ik->ij', b, coords)
    weights = np.c_[bary, 1 - bary.sum(axis=1)]
    
    nodes_tri = tri.simplices[valid_simplices]
    rows = np.repeat(np.where(valid)[0], 3)
    cols = nodes_tri.flatten()
    vals = weights.flatten()
    
    mesh2grid = sp.coo_matrix((vals, (rows, cols)), shape=(len(grid_points), mesh.nodes.shape[0]))
    
    grid_info = {
        'xgrid': xgrid,
        'ygrid': ygrid,
        'shape': X.shape,
        'mesh2grid': mesh2grid,
        'valid_mask': valid,
        'pixel_area': grid_step * grid_step
    }
    
    # Store grid info in meshes just in case
    mesh.vol = grid_info
    mesh_anomaly.vol = grid_info
    
    return mesh, mesh_anomaly, grid_info

def forward_operator(mesh):
    """
    Calculates the Jacobian (Sensitivity Matrix) and the forward data (Amplitude).
    Returns: (J, data_amp)
    """
    # 1. Forward Solver
    sys_matrix = assemble_fem(mesh, omega=0)
    Q = get_source_vector(mesh)
    
    solver = spla.factorized(sys_matrix)
    phi = solver(Q.toarray()) # (nnodes, nsources)
    
    # Extract Data (Amplitude)
    active_link_idx = np.where(mesh.link[:, 2] == 1)[0]
    links = mesh.link[active_link_idx]
    
    # Map sources to columns
    active_src_list = np.unique(links[:, 0]).astype(int)
    src_map = {src_idx: i for i, src_idx in enumerate(active_src_list)}
    
    det_int_func = mesh.meas['int_func']
    data_amp = np.zeros(len(links))
    
    for i, link in enumerate(links):
        src_idx = int(link[0])
        det_idx = int(link[1])
        col_idx = src_map[src_idx]
        field = phi[:, col_idx]
        
        det_entry = det_int_func[det_idx]
        elem_idx = int(det_entry[0])
        w = det_entry[1:]
        nodes_el = mesh.elements[elem_idx]
        val = np.sum(field[nodes_el] * w)
        data_amp[i] = np.abs(val)
        
    # 2. Adjoint Solver for Jacobian
    active_det_idx = np.unique(links[:, 1]).astype(int)
    n_adj_sources = len(active_det_idx)
    nnodes = mesh.nodes.shape[0]
    
    Q_adj = sp.lil_matrix((nnodes, n_adj_sources), dtype=complex)
    
    for i, det_idx in enumerate(active_det_idx):
        entry = det_int_func[det_idx]
        elem_idx = int(entry[0])
        w = entry[1:]
        nodes_el = mesh.elements[elem_idx]
        for j, n_idx in enumerate(nodes_el):
            Q_adj[n_idx, i] = w[j]
            
    phi_adj = solver(Q_adj.toarray())
    
    # 3. Calculate Jacobian on Grid
    M2G = mesh.vol['mesh2grid']
    phi_grid = M2G @ phi 
    phi_adj_grid = M2G @ phi_adj
    
    det_map = {det_idx: i for i, det_idx in enumerate(active_det_idx)}
    n_links = len(links)
    n_grid = M2G.shape[0]
    J = np.zeros((n_links, n_grid))
    
    # Calculate I0 for normalization
    # I0 is amplitude at detector. We already have data_amp.
    I0 = data_amp
    
    for i in range(n_links):
        src_idx = int(links[i, 0])
        det_idx = int(links[i, 1])
        
        s_col = src_map[src_idx]
        d_col = det_map[det_idx]
        
        # J = - phi * phi_adj * PixelArea
        # Negative sign due to diffusion definition of absorption perturbation
        # Real part usually taken for CW/Amplitude
        sens = - np.real(phi_grid[:, s_col] * phi_adj_grid[:, d_col])
        J[i, :] = sens
        
    J = J * mesh.vol['pixel_area']
    
    # Normalize by Intensity for Rytov/log-ratio compatibility: dOD ~ dI/I -> J / I
    # Avoid division by zero
    I0_safe = I0.copy()
    I0_safe[I0_safe == 0] = 1e-15
    J = J / I0_safe[:, None]
    
    return J, data_amp

def run_inversion(J, dOD, reg_factor=2.0):
    """
    Solves the inverse problem using Tikhonov regularization.
    min ||J*x - dOD||^2 + ||reg*x||^2
    """
    # Heuristic regularization scaling
    reg_val = reg_factor * np.max(np.abs(J))
    
    Gamma = reg_val
    ATA = J.T @ J
    n_params = J.shape[1]
    
    lhs = ATA + (Gamma**2) * np.eye(n_params)
    rhs = J.T @ dOD
    
    # Solve linear system
    # Use lstsq or solve. Since lhs is positive definite (ATA + reg^2 I), solve is fine.
    # For large systems, sparse solver or CG would be better, but grid here is manageable (45x45=2025).
    x = np.linalg.solve(lhs, rhs)
    
    return x

def evaluate_results(recon_vec, mesh_truth, mesh_baseline, grid_info, out_name='reconstruction_result.png'):
    """
    Computes metrics and plots results.
    """
    grid_shape = grid_info['shape']
    xgrid = grid_info['xgrid']
    ygrid = grid_info['ygrid']
    
    # Reconstruct Image
    recon_img = recon_vec.reshape(grid_shape)
    
    # Ground Truth Image
    dmua_mesh = mesh_truth.mua - mesh_baseline.mua
    dmua_truth_grid = grid_info['mesh2grid'] @ dmua_mesh
    truth_img = dmua_truth_grid.reshape(grid_shape)
    
    # Metrics
    mse = np.mean((recon_vec - dmua_truth_grid)**2)
    max_val = np.max(np.abs(dmua_truth_grid))
    if max_val == 0: max_val = 1.0
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-16))
    
    print(f"Evaluation Metrics:")
    print(f"  MSE: {mse:.6e}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    
    vmax = np.max(np.abs(truth_img))
    if vmax == 0: vmax = 1.0
    
    im1 = ax1.imshow(truth_img, origin='lower', extent=extent, cmap='jet', vmin=-vmax, vmax=vmax)
    ax1.set_title('Ground Truth ($\Delta \mu_a$)')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(recon_img, origin='lower', extent=extent, cmap='jet', vmin=-vmax, vmax=vmax)
    ax2.set_title(f'Reconstruction (PSNR: {psnr:.2f} dB)')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle("NIRFASTer Reconstruction Demo")
    plt.tight_layout()
    plt.savefig(out_name)
    print(f"Plot saved to {out_name}")
    
    return mse, psnr

if __name__ == '__main__':
    # --- Configuration ---
    MESH_FILE = 'meshes/standard/circle2000_86_stnd'
    GRID_STEP = 2.0
    ANOMALY_CENTER = [15.0, 15.0]
    ANOMALY_RADIUS = 10.0
    ANOMALY_FACTOR = 1.2
    
    # 1. Load Data
    print("Loading and preprocessing data...")
    mesh_base, mesh_anom, grid_info = load_and_preprocess_data(
        MESH_FILE, 
        anomaly_center=ANOMALY_CENTER, 
        anomaly_radius=ANOMALY_RADIUS, 
        anomaly_factor=ANOMALY_FACTOR, 
        grid_step=GRID_STEP
    )
    
    # 2. Forward Operator (Jacobian from Base, Data from Both)
    print("Running forward operator...")
    # We need Jacobian of the baseline
    J_base, data_base = forward_operator(mesh_base)
    
    # We need Data of the anomaly (simulation)
    # We do NOT need Jacobian of anomaly for linear reconstruction, just data.
    # But forward_operator calculates both. We ignore J_anom.
    _, data_anom = forward_operator(mesh_anom)
    
    # Calculate Data Difference (Log Ratio / Optical Density)
    # dOD = ln(I_anom) - ln(I_base) ??? 
    # Convention: OD = -ln(I/I0). dOD = OD_anom - OD_base = -ln(I_anom) + ln(I_base) = -ln(I_anom/I_base).
    # However, perturbation theory often uses ln(I_meas) - ln(I_ref) = J * dx. 
    # Let's use dOD = ln(I_anom) - ln(I_base).
    # If I_anom < I_base (absorber), dOD is negative.
    # But Jacobian for absorption is negative (-phi*phi_adj).
    # So negative J * positive dx = negative dOD. Consistent.
    
    dOD = np.log(data_anom + 1e-20) - np.log(data_base + 1e-20)
    
    # Add noise
    noise_level = 0.01 # 1% noise
    noise = np.random.randn(len(dOD)) * noise_level * np.max(np.abs(dOD))
    dOD_noisy = dOD + noise
    
    # 3. Inversion
    print("Running inversion...")
    recon_x = run_inversion(J_base, dOD_noisy, reg_factor=2.0)
    
    # 4. Evaluation
    print("Evaluating results...")
    evaluate_results(recon_x, mesh_anom, mesh_base, grid_info)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")