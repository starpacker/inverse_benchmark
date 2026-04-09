import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def project_points(particles_3d, camera, image_w, image_h):
    """
    Project 3D points onto a camera's image plane.
    """
    P = camera['P']
    N = particles_3d.shape[0]

    X_hom = np.hstack([particles_3d, np.ones((N, 1))])
    proj = (P @ X_hom.T).T

    depth = proj[:, 2]
    uv = proj[:, :2] / depth[:, np.newaxis]

    visible = (
        (depth > 0) &
        (uv[:, 0] >= 0) & (uv[:, 0] < image_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < image_h)
    )

    return uv, visible

def forward_operator(particles_3d, cameras, config, noise_std=0.5):
    """
    Full forward model: project all particles onto all cameras
    and add Gaussian detection noise.
    
    Forward model: Given 3D particle positions X_j, project onto each camera i
    via pinhole model: x_ij = K_i @ [R_i | t_i] @ X_j
    
    Args:
        particles_3d: (N, 3) array of 3D particle positions
        cameras: List of camera dictionaries
        config: Configuration dictionary with image dimensions
        noise_std: Standard deviation of detection noise [pixels]
    
    Returns:
        detections: list of N_CAMERAS arrays, each (N, 2) pixel coords
        visibility: list of N_CAMERAS boolean masks, each (N,)
        clean_projections: list of noise-free projections
    """
    image_w = config['image_w']
    image_h = config['image_h']
    
    detections = []
    visibility = []
    clean_projections = []

    for cam in cameras:
        uv_clean, vis = project_points(particles_3d, cam, image_w, image_h)

        # Add Gaussian noise to detections
        noise = np.random.normal(0, noise_std, uv_clean.shape)
        uv_noisy = uv_clean + noise

        # Re-check bounds after noise
        vis_noisy = (
            vis &
            (uv_noisy[:, 0] >= 0) & (uv_noisy[:, 0] < image_w) &
            (uv_noisy[:, 1] >= 0) & (uv_noisy[:, 1] < image_h)
        )

        detections.append(uv_noisy)
        visibility.append(vis_noisy)
        clean_projections.append(uv_clean)

    return detections, visibility, clean_projections
