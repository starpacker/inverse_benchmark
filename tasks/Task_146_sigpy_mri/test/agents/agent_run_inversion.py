import matplotlib

matplotlib.use('Agg')

import os

import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, 'repo')

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

import sigpy as sp

import sigpy.mri.app as mri_app

def run_inversion(kspace, mps, lamda=0.001, max_iter=200, wavelet='db4'):
    """
    Run the L1-Wavelet compressed sensing reconstruction.
    
    Solves the optimization problem:
        min_x  0.5 * ||M*F*S*x - y||_2^2  +  lambda * ||W*x||_1
    where W is the wavelet transform.
    
    Also computes zero-filled reconstruction as baseline.
    
    Args:
        kspace: undersampled k-space (num_coils, ny, nx)
        mps: sensitivity maps (num_coils, ny, nx)
        lamda: regularization parameter for L1 term
        max_iter: maximum number of iterations
        wavelet: wavelet name for transform (e.g., 'db4')
        
    Returns:
        dict containing:
            'recon': L1-Wavelet reconstructed image (ny, nx)
            'zero_filled': zero-filled reconstruction (ny, nx)
            'params': dictionary of reconstruction parameters
    """
    num_coils = kspace.shape[0]
    ny, nx = kspace.shape[1], kspace.shape[2]
    
    # Zero-filled (adjoint) reconstruction
    coil_imgs = np.zeros_like(kspace)
    for c in range(num_coils):
        coil_imgs[c] = sp.ifft(kspace[c], axes=(-2, -1))
    
    # Sensitivity-weighted combination
    zf_recon = np.zeros((ny, nx), dtype=np.complex64)
    for c in range(num_coils):
        zf_recon += np.conj(mps[c]) * coil_imgs[c]
    
    # Normalize by sum of squared sensitivities
    sens_norm = np.sum(np.abs(mps) ** 2, axis=0)
    sens_norm = np.maximum(sens_norm, 1e-8)
    zf_recon = zf_recon / sens_norm
    
    # L1-Wavelet compressed sensing reconstruction using sigpy
    recon = mri_app.L1WaveletRecon(
        kspace,
        mps,
        lamda=lamda,
        wave_name=wavelet,
        max_iter=max_iter,
        show_pbar=True,
    ).run()
    
    params = {
        'lamda': lamda,
        'max_iter': max_iter,
        'wavelet': wavelet,
        'method': 'L1-Wavelet Compressed Sensing'
    }
    
    return {
        'recon': recon,
        'zero_filled': zf_recon,
        'params': params
    }
