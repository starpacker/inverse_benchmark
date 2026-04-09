import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.special import hankel2

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_95_eispy2d"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def green2d(k0, r):
    """G(r) = (j/4) H_0^{(2)}(k0 r) – 2-D scalar Green's function."""
    r_safe = np.where(r < 1e-12, 1e-12, r)
    return (1j / 4.0) * hankel2(0, k0 * r_safe)

def run_inversion(data, lambda_reg="auto"):
    """
    Run Tikhonov inversion to reconstruct dielectric contrast.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data containing:
        - y_noisy: noisy measurements
        - k0, gx, gy, tx_angles, rx_pos, ds: forward model parameters
        - snr_db: SNR for auto-lambda computation
        - n_grid: grid size
    lambda_reg : float or str
        Tikhonov regularization parameter, or "auto" for automatic selection
        
    Returns
    -------
    chi_rec : ndarray
        Reconstructed dielectric contrast (n_grid, n_grid)
    lambda_used : float
        The regularization parameter that was used
    """
    y_noisy = data['y_noisy']
    k0 = data['k0']
    gx = data['gx']
    gy = data['gy']
    tx_angles = data['tx_angles']
    rx_pos = data['rx_pos']
    ds = data['ds']
    snr_db = data['snr_db']
    n_grid = data['n_grid']
    
    # Build sensing matrix
    Xg, Yg = np.meshgrid(gx, gy)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    n_pix = pts.shape[0]
    n_tx = len(tx_angles)
    n_rx = rx_pos.shape[0]
    
    A = np.zeros((n_tx * n_rx, n_pix), dtype=np.complex128)
    
    for l, theta in enumerate(tx_angles):
        d_hat = np.array([np.cos(theta), np.sin(theta)])
        E_inc = np.exp(1j * k0 * (pts @ d_hat))
        
        for m in range(n_rx):
            dr = np.sqrt((rx_pos[m, 0] - pts[:, 0]) ** 2
                         + (rx_pos[m, 1] - pts[:, 1]) ** 2)
            G = green2d(k0, dr)
            A[l * n_rx + m, :] = k0 ** 2 * G * E_inc * ds
    
    print(f"  A shape = {A.shape}")
    
    # Tikhonov inversion via SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    print(f"  SVD: {len(s)} singular values,  s_max={s[0]:.4e},  s_min={s[-1]:.4e}")
    
    if lambda_reg == "auto":
        noise_floor = s[0] * 10**(-snr_db / 20)
        lam = noise_floor
        print(f"  Noise floor = {noise_floor:.4e}")
        print(f"  Auto-lambda = {lam:.4e}")
        n_effective = np.sum(s > lam)
        print(f"  Effective rank = {n_effective}/{len(s)}")
    else:
        lam = lambda_reg
    
    # Tikhonov filter factors
    filt = s / (s ** 2 + lam ** 2)
    chi_rec_flat = Vh.conj().T @ (filt * (U.conj().T @ y_noisy))
    chi_rec = chi_rec_flat.real.reshape(n_grid, n_grid)
    
    print(f"  Lambda used = {lam:.4e}")
    
    # Clip to physical range
    chi_rec = np.clip(chi_rec, 0.0, None)
    print(f"  χ̂ range = [{chi_rec.min():.4f}, {chi_rec.max():.4f}]")
    
    # Save sensing matrix
    np.save(os.path.join(RESULTS_DIR, "sensing_matrix.npy"), A)
    
    return chi_rec, lam
