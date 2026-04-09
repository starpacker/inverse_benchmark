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
