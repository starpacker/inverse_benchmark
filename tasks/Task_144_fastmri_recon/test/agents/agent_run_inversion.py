import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from scipy.fft import fft2, ifft2, fftshift, ifftshift

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def total_variation(x):
    """Compute Total Variation of image x."""
    dx = np.diff(x, axis=1)
    dy = np.diff(x, axis=0)
    tv = np.sum(np.abs(dx)) + np.sum(np.abs(dy))
    return tv

def prox_tv(x, lam, n_iter=10):
    """
    Proximal operator for Total Variation using Chambolle's algorithm.
    """
    ny, nx = x.shape
    p1 = np.zeros((ny, nx), dtype=np.float64)
    p2 = np.zeros((ny, nx), dtype=np.float64)
    tau = 0.25

    for _ in range(n_iter):
        div_p = np.zeros_like(x)
        div_p[:, 1:] += p1[:, 1:]
        div_p[:, 0] += p1[:, 0]
        div_p[:, :-1] -= p1[:, :-1]

        div_p[1:, :] += p2[1:, :]
        div_p[0, :] += p2[0, :]
        div_p[:-1, :] -= p2[:-1, :]

        u = x - lam * div_p
        gx = np.zeros_like(u)
        gy = np.zeros_like(u)
        gx[:, :-1] = u[:, 1:] - u[:, :-1]
        gy[:-1, :] = u[1:, :] - u[:-1, :]

        norm_g = np.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        p1 = (p1 + tau * gx) / (1.0 + tau * norm_g)
        p2 = (p2 + tau * gy) / (1.0 + tau * norm_g)

    div_p = np.zeros_like(x)
    div_p[:, 1:] += p1[:, 1:]
    div_p[:, 0] += p1[:, 0]
    div_p[:, :-1] -= p1[:, :-1]
    div_p[1:, :] += p2[1:, :]
    div_p[0, :] += p2[0, :]
    div_p[:-1, :] -= p2[:-1, :]

    return x - lam * div_p

def run_inversion(kspace_undersampled, mask, n_iter=200, step_size=0.5,
                  lam=0.005, tv_inner_iter=20, verbose=True):
    """
    Run compressed sensing MRI reconstruction using ISTA with TV regularization.
    
    Minimizes: (1/2)||F_u x - y||^2 + lam * TV(x)
    
    Also computes zero-filled baseline reconstruction.
    
    Args:
        kspace_undersampled: undersampled k-space data (complex numpy array)
        mask: undersampling mask (numpy array)
        n_iter: number of ISTA iterations
        step_size: gradient descent step size
        lam: TV regularization weight
        tv_inner_iter: iterations for TV proximal operator
        verbose: print progress
    
    Returns:
        result: dictionary containing:
            - 'recon_cs': CS-TV reconstruction (numpy array)
            - 'recon_zf': Zero-filled reconstruction (numpy array)
    """
    print("\n[run_inversion] Zero-filled IFFT reconstruction (baseline)...")
    recon_zf = np.abs(fftshift(ifft2(ifftshift(kspace_undersampled))))
    
    print("[run_inversion] ISTA-TV compressed sensing reconstruction...")
    x = recon_zf.copy()
    
    for i in range(n_iter):
        # Gradient of data fidelity: F_u^H (F_u x - y)
        kspace_x = fftshift(fft2(ifftshift(x)))
        residual = mask * kspace_x - kspace_undersampled
        grad = np.real(fftshift(ifft2(ifftshift(mask * residual))))
        
        # Gradient step
        x_temp = x - step_size * grad
        
        # Proximal step (TV regularization)
        x = prox_tv(x_temp, lam=lam * step_size, n_iter=tv_inner_iter)
        
        # Enforce non-negativity
        x = np.clip(x, 0, None)
        
        if verbose and (i + 1) % 50 == 0:
            kspace_x = fftshift(fft2(ifftshift(x)))
            residual = mask * kspace_x - kspace_undersampled
            data_fit = 0.5 * np.sum(np.abs(residual) ** 2)
            tv_val = total_variation(x)
            print(f"  Iter {i+1}/{n_iter}: data_fit={data_fit:.4f}, TV={tv_val:.4f}")
    
    recon_cs = x
    
    result = {
        'recon_cs': recon_cs,
        'recon_zf': recon_zf
    }
    
    return result
