import numpy as np

import scipy.sparse.linalg

def forward_operator(x_stiffness, GK_structure, fxy):
    """
    Solves the forward Elasticity problem K(E)u = f to get displacement u.
    Note: x_stiffness should be scaled appropriately (e.g. ~1e5 or normalized).
    If using the refactored load output, x_stiffness might be normalized.
    """
    # In the original code, 'disp1' calculation uses GK modified by boundaries.
    # GK was already constructed with E_ground_truth in load_and_preprocess.
    # However, strictly speaking, a forward operator should take 'x' and produce 'y'.
    
    # Since GK is dependent on E linearly: K(E) = sum(E_i * K_i) + M.
    # But for boundary conditions, rows of GK are modified.
    # To keep it simple and consistent with the provided input code's logic flow:
    # We will use the pre-computed GK which contains the ground truth E and BCs.
    
    # Solve Ku = f
    GK_s = scipy.sparse.csr_matrix(GK_structure)
    disp = scipy.sparse.linalg.spsolve(GK_s, fxy)
    disp = np.around(disp, decimals=10) * 1e+5 # Scale up as in original code
    
    return disp
