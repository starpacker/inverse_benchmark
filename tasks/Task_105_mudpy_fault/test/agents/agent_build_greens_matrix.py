import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def okada_greens_function(obs_x, obs_y, patch_cx, patch_cy, patch_depth,
                          patch_length, patch_width, dip_rad, poisson):
    """
    Okada (1985) — simplified closed-form for a point dislocation
    (reverse/thrust fault in elastic half-space).
    
    Uses the Mogi-Okada point-source approximation for dip-slip faults.
    Returns (ux, uy, uz) surface displacement per unit slip.
    """
    dx = obs_x - patch_cx
    dy = obs_y - patch_cy
    d = patch_depth

    cos_dip = np.cos(dip_rad)
    sin_dip = np.sin(dip_rad)

    A = patch_length * patch_width

    R = np.sqrt(dx**2 + dy**2 + d**2)
    R3 = R**3
    R5 = R**5

    if R < 0.01:
        return 0.0, 0.0, 0.0

    factor = A / (4.0 * np.pi)

    ux = factor * (3.0 * dx * d * sin_dip / R5 -
                   (1.0 - 2.0 * poisson) * dx * sin_dip / R3)
    uy = factor * (3.0 * dy * d * sin_dip / R5 -
                   (1.0 - 2.0 * poisson) * dy * sin_dip / R3)
    uz = factor * (3.0 * d**2 * sin_dip / R5 +
                   (1.0 - 2.0 * poisson) * sin_dip / R3 +
                   cos_dip * d / R3)

    return ux, uy, uz

def build_greens_matrix(obs_coords, patch_params, poisson):
    """
    Build the Green's function matrix G such that d = G * s.
    d: displacement vector (3*N_obs,)
    s: slip vector (N_patches,)
    G: Green's function matrix (3*N_obs, N_patches)
    """
    n_obs = obs_coords.shape[0]
    n_patches = len(patch_params)
    G = np.zeros((3 * n_obs, n_patches))

    for j, patch in enumerate(patch_params):
        for i in range(n_obs):
            ux, uy, uz = okada_greens_function(
                obs_coords[i, 0], obs_coords[i, 1],
                patch["cx"], patch["cy"], patch["depth"],
                patch["length"], patch["width"], patch["dip_rad"],
                poisson
            )
            G[3 * i, j] = ux
            G[3 * i + 1, j] = uy
            G[3 * i + 2, j] = uz

    return G
