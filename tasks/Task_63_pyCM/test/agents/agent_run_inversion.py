import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import minimize_scalar

from scipy.linalg import solve, eigh

from scipy.sparse import diags

from numpy import kron

def run_inversion(C, disp_meas, nx, ny):
    """
    Tikhonov-regularised least-squares inversion with GCV
    for optimal regularisation parameter selection.
    
    min ||C·σ - u||² + λ||L·σ||²
    
    where L is the discrete Laplacian (smoothness prior).
    
    Parameters
    ----------
    C : ndarray (n, n)       Influence matrix.
    disp_meas : ndarray (n,) Measured displacement [mm].
    nx, ny : int             Grid dimensions.
    
    Returns
    -------
    stress_rec : ndarray (n,)  Reconstructed stress [MPa].
    """
    print("[RECON] Tikhonov inversion with GCV ...")
    n = C.shape[1]
    
    # Smoothness regularisation (2D Laplacian)
    D1x = diags([-1, 1], [0, 1], shape=(nx-1, nx)).toarray()
    D1y = diags([-1, 1], [0, 1], shape=(ny-1, ny)).toarray()
    Lx = kron(np.eye(ny), D1x.T @ D1x)
    Ly = kron(D1y.T @ D1y, np.eye(nx))
    L = Lx + Ly
    
    CtC = C.T @ C
    Ctd = C.T @ disp_meas
    
    # SVD-based GCV for numerical stability
    # Regularise L slightly for positive-definiteness
    L_reg = L + 1e-12 * np.eye(n)
    eigvals, V = eigh(CtC, L_reg)  # CtC v = eigval * L_reg v
    # Transform data
    Ctd_t = V.T @ Ctd
    
    # GCV criterion using filter factors
    def gcv(log_lam):
        lam = 10**log_lam
        filt = eigvals / (eigvals + lam)  # filter factors
        sigma_t2 = Ctd_t / (eigvals + lam)
        stress_v = V @ sigma_t2
        resid = C @ stress_v - disp_meas
        trH = np.sum(filt)  # trace of hat matrix = sum of filter factors
        nn = len(disp_meas)
        denom = max((1 - trH / nn), 1e-6) ** 2
        return np.sum(resid**2) / nn / denom
    
    # Search for optimal lambda — use tighter bounds for low-noise problem
    result = minimize_scalar(gcv, bounds=(-10, 2), method='bounded',
                             options={'xatol': 1e-4})
    best_lam = 10**result.x
    print(f"[RECON]   Optimal λ = {best_lam:.2e} (GCV)")
    
    # Also try a grid search to avoid local minima
    log_lams = np.linspace(-10, 2, 200)
    gcv_vals = [gcv(ll) for ll in log_lams]
    best_idx = np.argmin(gcv_vals)
    best_lam_grid = 10**log_lams[best_idx]
    print(f"[RECON]   Grid-search λ = {best_lam_grid:.2e}")
    
    # Use whichever gives lower GCV
    if gcv_vals[best_idx] < gcv(np.log10(best_lam)):
        best_lam = best_lam_grid
        print(f"[RECON]   Using grid-search λ = {best_lam:.2e}")
    
    A = CtC + best_lam * L
    stress_rec = solve(A, Ctd, assume_a='pos')
    
    return stress_rec
