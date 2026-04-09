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
