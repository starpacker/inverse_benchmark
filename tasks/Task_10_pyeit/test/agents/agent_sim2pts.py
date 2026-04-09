import numpy as np


# --- Extracted Dependencies ---

def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    """Interpolate element values to node values for visualization"""
    n_nodes = pts.shape[0]
    node_val = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    
    for i, el in enumerate(sim):
        node_val[el] += sim_values[i]
        count[el] += 1
        
    return node_val / np.maximum(count, 1)
