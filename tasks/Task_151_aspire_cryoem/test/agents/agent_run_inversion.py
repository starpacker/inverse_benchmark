import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from scipy.ndimage import affine_transform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def reconstruct_mean_estimator(sim):
    """
    Reconstruct volume using ASPIRE's MeanEstimator.
    """
    from aspire.reconstruction import MeanEstimator

    print("  Running ASPIRE MeanEstimator...")
    estimator = MeanEstimator(sim)
    recon_vol = estimator.estimate()
    recon_data = recon_vol.asnumpy().squeeze()
    print(f"  MeanEstimator reconstruction shape: {recon_data.shape}")
    return recon_data

def reconstruct_back_projection(images, rotations, L):
    """
    Weighted back-projection reconstruction.
    """
    print("  Running Weighted Back-Projection reconstruction...")
    N = images.shape[0]

    recon = np.zeros((L, L, L), dtype=np.float64)

    for i in range(N):
        # Expand 2D projection into 3D by repeating along z
        proj_3d = np.repeat(images[i][np.newaxis, :, :], L, axis=0)

        # Inverse rotation: rotate the back-projected 3D image
        R_inv = rotations[i].T

        # Use affine_transform with the rotation matrix
        c = np.array([(L - 1) / 2.0] * 3)
        offset = c - R_inv @ c

        rotated_bp = affine_transform(
            proj_3d, R_inv, offset=offset,
            order=1, mode='constant', cval=0.0
        )
        recon += rotated_bp

    recon /= N

    print(f"  Weighted Back-Projection reconstruction done.")
    return recon

def run_inversion(data_dict, method='mean_estimator'):
    """
    Run 3D volume reconstruction from 2D projections.
    
    Implements inverse reconstruction using either ASPIRE's MeanEstimator
    or weighted back-projection algorithm.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data containing:
            - noisy_images: Noisy projection images (N, L, L)
            - rotations: Rotation matrices (N, 3, 3)
            - sim: ASPIRE Simulation object
            - vol_size: Volume side length
        method: Reconstruction method ('mean_estimator' or 'back_projection')
    
    Returns:
        Dictionary containing:
            - recon_volume: Reconstructed 3D volume (L, L, L)
            - method: Method name used
            - success: Boolean indicating success
    """
    noisy_images = data_dict['noisy_images']
    rotations = data_dict['rotations']
    sim = data_dict['sim']
    vol_size = data_dict['vol_size']

    result = {
        'method': method,
        'success': False,
        'recon_volume': None,
    }

    if method == 'mean_estimator':
        print("\n[3/5] Reconstructing volume with ASPIRE MeanEstimator...")
        try:
            recon = reconstruct_mean_estimator(sim)
            result['recon_volume'] = recon
            result['success'] = True
        except Exception as e:
            print(f"  MeanEstimator failed: {e}")
    elif method == 'back_projection':
        print("\n[4/5] Reconstructing volume with weighted back-projection...")
        try:
            recon = reconstruct_back_projection(noisy_images, rotations, vol_size)
            result['recon_volume'] = recon
            result['success'] = True
        except Exception as e:
            print(f"  BackProjection failed: {e}")
    else:
        raise ValueError(f"Unknown method: {method}")

    return result
