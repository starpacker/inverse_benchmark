import numpy as np

import matplotlib

matplotlib.use('Agg')

def trilinear_interp(vol3d, coords, N):
    """Trilinear interpolation for extracting values from 3D volume."""
    coords = np.clip(coords, 0, N - 1.001)
    x0 = np.floor(coords[:, 0]).astype(int)
    y0 = np.floor(coords[:, 1]).astype(int)
    z0 = np.floor(coords[:, 2]).astype(int)
    x1 = np.minimum(x0 + 1, N - 1)
    y1 = np.minimum(y0 + 1, N - 1)
    z1 = np.minimum(z0 + 1, N - 1)
    xd = coords[:, 0] - x0
    yd = coords[:, 1] - y0
    zd = coords[:, 2] - z0

    c000 = vol3d[z0, y0, x0]
    c001 = vol3d[z0, y0, x1]
    c010 = vol3d[z0, y1, x0]
    c011 = vol3d[z0, y1, x1]
    c100 = vol3d[z1, y0, x0]
    c101 = vol3d[z1, y0, x1]
    c110 = vol3d[z1, y1, x0]
    c111 = vol3d[z1, y1, x1]

    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd
