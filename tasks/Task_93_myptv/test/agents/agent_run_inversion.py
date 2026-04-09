import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def triangulate_dlt(uv_list, P_list):
    """
    Triangulate a single 3D point from its 2D projections
    in multiple cameras using Direct Linear Transform (DLT).
    """
    n_views = len(uv_list)
    A = np.zeros((2 * n_views, 4))

    for i, (uv, P) in enumerate(zip(uv_list, P_list)):
        u, v = uv
        A[2 * i] = u * P[2, :] - P[0, :]
        A[2 * i + 1] = v * P[2, :] - P[1, :]

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1, :]

    X_3d = X_hom[:3] / X_hom[3]
    return X_3d

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

def run_inversion(detections, visibility, cameras):
    """
    Reconstruct 3D particle positions from multi-camera 2D detections.
    
    Inverse problem: Given 2D detections {x_ij} across cameras, triangulate
    3D positions using Direct Linear Transform (DLT) / least-squares ray intersection.
    
    For each particle visible in ≥2 cameras, gather its detections
    and triangulate via DLT.
    
    Args:
        detections: list of (N, 2) arrays per camera
        visibility: list of (N,) boolean masks per camera
        cameras: list of camera dicts
    
    Returns:
        recon_3d: (N, 3) array of reconstructed positions (NaN for unrecoverable)
        n_views_per_particle: (N,) number of views used
        reproj_errors: (N,) reprojection errors
    """
    N = detections[0].shape[0]
    recon_3d = np.full((N, 3), np.nan)
    n_views = np.zeros(N, dtype=int)
    reproj_errors = np.full(N, np.nan)

    for j in range(N):
        # Gather detections for particle j across all cameras
        uv_list = []
        P_list = []

        for i, cam in enumerate(cameras):
            if visibility[i][j]:
                uv_list.append(detections[i][j])
                P_list.append(cam['P'])

        n_views[j] = len(uv_list)

        if len(uv_list) >= 2:
            X_3d = triangulate_dlt(uv_list, P_list)
            recon_3d[j] = X_3d
            reproj_errors[j] = compute_reprojection_error(X_3d, uv_list, P_list)

    return recon_3d, n_views, reproj_errors
