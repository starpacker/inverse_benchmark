import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from scipy.ndimage import affine_transform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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
