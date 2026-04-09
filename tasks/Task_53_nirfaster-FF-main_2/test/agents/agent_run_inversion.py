import numpy as np

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
