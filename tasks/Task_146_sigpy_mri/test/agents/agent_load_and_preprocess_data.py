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

import sigpy.mri as mri

def load_and_preprocess_data(img_shape, num_coils, accel_factor, noise_level=1e-4, seed=42):
    """
    Generate and preprocess all data needed for MRI reconstruction.
    
    This function generates:
    - Shepp-Logan phantom (ground truth image)
    - Birdcage coil sensitivity maps
    - Poisson variable-density undersampling mask
    - Simulated undersampled multi-coil k-space data
    
    Args:
        img_shape: tuple (ny, nx) image dimensions
        num_coils: number of receiver coils
        accel_factor: acceleration factor for undersampling
        noise_level: noise standard deviation
        seed: random seed for reproducibility
        
    Returns:
        dict containing:
            'phantom': ground truth image (ny, nx)
            'mps': sensitivity maps (num_coils, ny, nx)
            'mask': undersampling mask (ny, nx)
            'kspace': undersampled k-space (num_coils, ny, nx)
            'config': dictionary of configuration parameters
    """
    np.random.seed(seed)
    
    # Generate Shepp-Logan phantom normalized to [0, 1]
    phantom = sp.shepp_logan(img_shape)
    phantom = np.abs(phantom)
    phantom = phantom / phantom.max()
    phantom = phantom.astype(np.complex64)
    
    # Generate birdcage coil sensitivity maps
    mps = mri.birdcage_maps((num_coils,) + img_shape)
    mps = mps.astype(np.complex64)
    
    # Generate Poisson variable-density undersampling mask
    mask = mri.poisson(img_shape, accel=accel_factor, seed=42)
    mask = mask.astype(np.complex64)
    
    # Simulate undersampled multi-coil k-space: y = M * F * S * x + noise
    ny, nx = img_shape
    kspace = np.zeros((num_coils, ny, nx), dtype=np.complex64)
    
    for c in range(num_coils):
        coil_img = mps[c] * phantom
        # Use sigpy FFT (centered, orthonormal normalization)
        coil_kspace = sp.fft(coil_img, axes=(-2, -1))
        kspace[c] = coil_kspace * mask
    
    # Add complex Gaussian noise only to sampled locations
    rng = np.random.RandomState(12345)
    noise = noise_level * (rng.randn(*kspace.shape) + 1j * rng.randn(*kspace.shape))
    kspace = kspace + (noise * mask[np.newaxis]).astype(np.complex64)
    
    # Compute sampling ratio
    sampling_ratio = np.sum(np.abs(mask) > 0) / mask.size
    
    config = {
        'img_shape': img_shape,
        'num_coils': num_coils,
        'accel_factor': accel_factor,
        'noise_level': noise_level,
        'sampling_ratio': sampling_ratio,
        'seed': seed
    }
    
    return {
        'phantom': phantom,
        'mps': mps,
        'mask': mask,
        'kspace': kspace,
        'config': config
    }
