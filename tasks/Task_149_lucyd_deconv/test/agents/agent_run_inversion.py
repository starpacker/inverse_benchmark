import os

import numpy as np

from scipy.signal import fftconvolve

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def forward_operator(x, psf):
    """
    Forward operator for fluorescence microscopy: convolution with PSF.
    
    Parameters
    ----------
    x : ndarray
        Input image (estimate)
    psf : ndarray
        Point spread function
        
    Returns
    -------
    y_pred : ndarray
        Predicted measurement (convolved image)
    """
    y_pred = fftconvolve(x, psf, mode='same')
    y_pred = np.clip(y_pred, 1e-12, None)
    return y_pred

def run_inversion(observed, psf, n_iterations=150, tv_weight=0.002):
    """
    Richardson-Lucy deconvolution for Poisson noise model with TV regularization.
    
    The RL update rule:
        x_{k+1} = x_k * [PSF^T ⊛ (y / (PSF ⊛ x_k))]
    
    Parameters
    ----------
    observed : ndarray
        Noisy blurred observation
    psf : ndarray
        Point spread function
    n_iterations : int
        Number of RL iterations
    tv_weight : float
        TV regularization weight
        
    Returns
    -------
    estimate : ndarray
        Deconvolved image
    """
    eps = 1e-12
    psf_flipped = psf[::-1, ::-1]

    # Initialize with the observed image
    estimate = np.copy(observed)
    estimate = np.clip(estimate, eps, None)

    for i in range(n_iterations):
        # Forward: convolve estimate with PSF
        est_conv = forward_operator(estimate, psf)

        # Compute ratio: y / (H * x_k)
        ratio = observed / est_conv

        # Correlate ratio with PSF (adjoint operation)
        correction = fftconvolve(ratio, psf_flipped, mode='same')

        # RL multiplicative update
        estimate = estimate * correction

        # TV regularization
        if tv_weight > 0:
            # Compute gradients
            dx = np.zeros_like(estimate)
            dy = np.zeros_like(estimate)
            dx[:-1, :] = np.diff(estimate, axis=0)
            dy[:, :-1] = np.diff(estimate, axis=1)

            # Gradient magnitude
            grad_mag = np.sqrt(dx**2 + dy**2 + eps)

            # Normalized gradients
            nx = dx / grad_mag
            ny = dy / grad_mag

            # Divergence of normalized gradient
            div = np.zeros_like(estimate)
            div[1:, :] += np.diff(nx, axis=0)
            div[:, 1:] += np.diff(ny, axis=1)

            # TV correction
            estimate = estimate + tv_weight * div
            estimate = np.clip(estimate, 0, None)

        # Enforce non-negativity
        estimate = np.clip(estimate, eps, None)

    # Normalize reconstruction to [0, 1]
    estimate = np.clip(estimate, 0, None)
    estimate = estimate / (estimate.max() + 1e-12)

    return estimate
