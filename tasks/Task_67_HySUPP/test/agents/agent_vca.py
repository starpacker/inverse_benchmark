import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def vca(Y, n_end, rng):
    """
    Vertex Component Analysis (VCA) for endmember extraction.
    """
    L, P = Y.shape
    
    Y_mean = Y.mean(axis=1, keepdims=True)
    Y_centered = Y - Y_mean
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    Ud = U[:, :n_end]
    Y_proj = Ud.T @ Y_centered
    
    best_indices = None
    best_volume = -1
    
    for trial in range(10):
        indices = []
        for i in range(n_end):
            if i == 0:
                w = rng.standard_normal(n_end)
            else:
                E_sel = Y_proj[:, indices]
                proj_matrix = E_sel @ np.linalg.pinv(E_sel)
                w = (np.eye(n_end) - proj_matrix) @ rng.standard_normal(n_end)
            
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-10:
                w /= w_norm
            
            projections = w @ Y_proj
            idx = np.argmax(np.abs(projections))
            if idx in indices:
                sorted_idx = np.argsort(np.abs(projections))[::-1]
                for candidate in sorted_idx:
                    if candidate not in indices:
                        idx = candidate
                        break
            indices.append(idx)
        
        E_trial = Y_proj[:, indices]
        try:
            vol = abs(np.linalg.det(E_trial))
        except:
            vol = 0
        
        if vol > best_volume:
            best_volume = vol
            best_indices = indices
    
    E_vca = Y[:, best_indices]
    print(f"  VCA selected pixel indices: {best_indices} (vol={best_volume:.4e})")
    return E_vca, best_indices
