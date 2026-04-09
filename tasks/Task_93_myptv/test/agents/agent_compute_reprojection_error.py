import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def compute_reprojection_error(X_3d, uv_list, P_list):
    """Compute mean reprojection error for a triangulated point."""
    X_hom = np.append(X_3d, 1.0)
    errors = []
    for uv, P in zip(uv_list, P_list):
        proj = P @ X_hom
        proj_2d = proj[:2] / proj[2]
        err = np.linalg.norm(proj_2d - uv)
        errors.append(err)
    return np.mean(errors)
