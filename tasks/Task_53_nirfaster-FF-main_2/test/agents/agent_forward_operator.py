import numpy as np

import scipy.sparse as sp

import scipy.sparse.linalg as spla

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
