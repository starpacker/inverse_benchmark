import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import shift as ndi_shift, uniform_filter, gaussian_filter

def load_and_preprocess_data(scene_size=128, n_angular=5, baseline=0.8, noise_std=0.005, seed=42):
    """
    Create a synthetic scene with depth map, render the light field,
    convert to raw MLA image, and add noise.
    
    Returns:
        dict containing:
            - gt_scene: ground truth scene (2D array)
            - gt_depth: ground truth depth map (2D array)
            - raw_noisy: noisy raw MLA sensor image (2D array)
            - params: dictionary of parameters
    """
    np.random.seed(seed)
    
    # Create scene and depth map
    size = scene_size
    scene = np.zeros((size, size), dtype=np.float64)
    depth = np.full((size, size), 5.0, dtype=np.float64)  # background depth

    # Background: smooth gradient
    yy, xx = np.mgrid[0:size, 0:size]
    scene = 0.3 + 0.3 * (xx / size) + 0.1 * np.sin(2 * np.pi * yy / size * 3)

    # Object 1: circle at depth Z=2.0 (near, large disparity)
    cy, cx, r = size // 3, size // 3, size // 6
    mask_circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    scene[mask_circle] = 0.85
    depth[mask_circle] = 2.0

    # Object 2: square at depth Z=3.5 (mid)
    sy, sx = int(size * 0.55), int(size * 0.55)
    half = size // 8
    mask_sq = (np.abs(yy - sy) < half) & (np.abs(xx - sx) < half)
    scene[mask_sq] = 0.55
    depth[mask_sq] = 3.5

    # Add mild texture so block matching has features to lock onto
    texture = 0.05 * np.random.RandomState(123).randn(size, size)
    texture = gaussian_filter(texture, sigma=1.0)
    scene = np.clip(scene + texture, 0, 1)

    gt_scene = scene.copy()
    gt_depth = depth.copy()

    # Render light field using forward operator
    lf = forward_operator(scene, depth, n_angular, baseline)

    # Convert light field to raw MLA sensor image
    n_ang = lf.shape[0]
    s_size = lf.shape[2]
    raw_h = n_ang * s_size
    raw_clean = np.zeros((raw_h, raw_h), dtype=np.float64)
    for u in range(n_ang):
        for v in range(n_ang):
            raw_clean[u::n_ang, v::n_ang] = lf[u, v]

    # Add Gaussian noise
    raw_noisy = raw_clean + noise_std * np.random.randn(*raw_clean.shape)
    raw_noisy = np.clip(raw_noisy, 0, 1)

    params = {
        'scene_size': scene_size,
        'n_angular': n_angular,
        'baseline': baseline,
        'noise_std': noise_std,
        'patch_half': 5,
        'disp_range': 6,
    }

    return {
        'gt_scene': gt_scene,
        'gt_depth': gt_depth,
        'raw_noisy': raw_noisy,
        'params': params,
    }

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
