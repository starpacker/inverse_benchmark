import numpy as np

import scipy.sparse as sp

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
