import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.ndimage import gaussian_filter

def build_convolution_matrix(wavelet, n):
    """
    Build the convolution matrix H such that H @ r ≈ fftconvolve(r, wavelet, 'same').
    """
    w_len = len(wavelet)
    half_w = w_len // 2
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(w_len):
            col = i - half_w + j
            if 0 <= col < n:
                H[i, col] = wavelet[j]
    return H

def run_inversion(bscan_noisy, wavelet, eps_surface=4.0, lam_reg=0.003, smooth_sigma=3.0):
    """
    Run GPR full-waveform inversion to recover permittivity from B-scan.
    
    Algorithm:
    1. Tikhonov-regularized deconvolution to recover reflectivity from each trace
    2. Impedance inversion to convert reflectivity to permittivity
    3. Gaussian smoothing to reduce noise artifacts
    
    Parameters:
    -----------
    bscan_noisy : ndarray
        Noisy B-scan data, shape (nz, nx)
    wavelet : ndarray
        Source wavelet (Ricker), shape (wavelet_pts,)
    eps_surface : float
        Known surface permittivity for impedance inversion
    lam_reg : float
        Tikhonov regularization parameter
    smooth_sigma : float
        Gaussian smoothing sigma for post-processing
    
    Returns:
    --------
    eps_recon : ndarray
        Reconstructed permittivity model, shape (nz, nx)
    """
    nz, nx = bscan_noisy.shape
    eps_recon = np.zeros((nz, nx))
    
    # Pre-compute convolution matrix and H^T H (shared across traces)
    H = build_convolution_matrix(wavelet, nz)
    HtH = H.T @ H
    
    for ix in range(nx):
        # Tikhonov-regularized deconvolution
        bscan_trace = bscan_noisy[:, ix]
        n = len(bscan_trace)
        I = np.eye(n)
        A = HtH + lam_reg * I
        r_est = np.linalg.solve(A, H.T @ bscan_trace)
        
        # Impedance inversion
        sqrt_eps = np.zeros(n)
        sqrt_eps[0] = np.sqrt(eps_surface)
        
        for i in range(n - 1):
            r = np.clip(r_est[i], -0.95, 0.95)
            sqrt_eps[i + 1] = sqrt_eps[i] * (1 + r) / (1 - r)
        
        # Ensure positivity
        sqrt_eps = np.clip(sqrt_eps, 0.1, 100.0)
        eps_recon[:, ix] = sqrt_eps ** 2
    
    # Smoothing to reduce noise artifacts
    eps_recon = gaussian_filter(eps_recon, sigma=smooth_sigma)
    
    return eps_recon
