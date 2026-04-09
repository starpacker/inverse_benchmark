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

def forward_operator(x, mps, mask):
    """
    Apply the forward MRI acquisition model: y = M * F * S * x
    
    This implements the multi-coil forward operator:
    - S: multiplication by coil sensitivity maps
    - F: 2D Fourier transform (sigpy convention)
    - M: undersampling mask application
    
    Args:
        x: image to transform (ny, nx) complex array
        mps: sensitivity maps (num_coils, ny, nx) complex array
        mask: undersampling mask (ny, nx) binary/complex array
        
    Returns:
        y_pred: predicted k-space measurements (num_coils, ny, nx)
    """
    num_coils = mps.shape[0]
    ny, nx = x.shape
    y_pred = np.zeros((num_coils, ny, nx), dtype=np.complex64)
    
    for c in range(num_coils):
        # Apply sensitivity map
        coil_img = mps[c] * x
        # Apply 2D FFT using sigpy convention
        coil_kspace = sp.fft(coil_img, axes=(-2, -1))
        # Apply undersampling mask
        y_pred[c] = coil_kspace * mask
    
    return y_pred
