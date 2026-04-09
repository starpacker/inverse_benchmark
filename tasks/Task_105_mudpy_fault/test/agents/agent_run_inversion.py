import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

import time

from scipy.optimize import nnls

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def build_laplacian(nx, ny):
    """Build 2D Laplacian smoothing matrix for fault patches."""
    n = nx * ny
    L = np.zeros((n, n))

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            count = 0

            if i > 0:
                L[idx, idx - 1] = -1.0
                count += 1
            if i < nx - 1:
                L[idx, idx + 1] = -1.0
                count += 1
            if j > 0:
                L[idx, idx - nx] = -1.0
                count += 1
            if j < ny - 1:
                L[idx, idx + nx] = -1.0
                count += 1

            L[idx, idx] = float(count)

    return L

def run_inversion(G, d_obs, nx, ny, lambda_reg):
    """
    Run Tikhonov-regularized non-negative least squares inversion.
    
    Solves: s_hat = argmin ||Gs - d||² + λ||∇s||² subject to s ≥ 0
    
    The regularization uses a 2D Laplacian operator to enforce smoothness
    in the slip distribution while allowing for spatial variations.
    
    Args:
        G: Green's function matrix (3*N_obs, N_patches)
        d_obs: observed displacement vector (3*N_obs,)
        nx: number of fault patches along strike
        ny: number of fault patches along dip
        lambda_reg: Tikhonov regularization parameter
    
    Returns:
        dict containing:
        - s_hat: recovered slip vector (N_patches,)
        - rec_slip: recovered slip as 2D array (ny, nx)
        - inversion_time: time taken for inversion
    """
    print(f"[6] Inverting for slip (λ={lambda_reg}) ...")
    t0 = time.time()

    L = build_laplacian(nx, ny)

    G_aug = np.vstack([G, np.sqrt(lambda_reg) * L])
    d_aug = np.concatenate([d_obs, np.zeros(nx * ny)])

    s_hat, residual = nnls(G_aug, d_aug)

    t_inv = time.time() - t0
    rec_slip = s_hat.reshape(ny, nx)

    print(f"    Inversion: {t_inv:.1f}s")
    print(f"    Reconstructed max slip: {rec_slip.max():.2f} m")

    result = {
        "s_hat": s_hat,
        "rec_slip": rec_slip,
        "inversion_time": t_inv,
        "residual": residual,
    }

    return result
