import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import shift as ndi_shift, uniform_filter, gaussian_filter

def forward_operator(scene, depth, n_angular=5, baseline=0.8):
    """
    Forward model: Render a 4-D light field L[u, v, s, t] from a 2-D scene + depth map.
    
    For sub-aperture (u, v), the scene is shifted by:
        dx = baseline * (u - u_c) / Z(s,t)
        dy = baseline * (v - v_c) / Z(s,t)
    
    Args:
        scene: 2D array of scene intensities
        depth: 2D array of depth values
        n_angular: number of angular samples per axis
        baseline: baseline parameter controlling disparity magnitude
    
    Returns:
        lf: 4D light field array of shape (n_angular, n_angular, size, size)
    """
    size = scene.shape[0]
    n_ang = n_angular
    u_c = (n_ang - 1) / 2.0
    v_c = u_c
    lf = np.zeros((n_ang, n_ang, size, size), dtype=np.float64)

    for u in range(n_ang):
        for v in range(n_ang):
            du = baseline * (u - u_c)
            dv = baseline * (v - v_c)

            # Approximate: use mean disparity per depth layer for efficiency
            shifted = np.zeros_like(scene)
            for z_val in np.unique(depth):
                mask = depth == z_val
                dx = du / z_val
                dy = dv / z_val
                layer = np.where(mask, scene, 0.0)
                shifted_layer = ndi_shift(layer, [dy, dx], order=1, mode='nearest')
                shifted += shifted_layer

            lf[u, v] = shifted

    return lf
