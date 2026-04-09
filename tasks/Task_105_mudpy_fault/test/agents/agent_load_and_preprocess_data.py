import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

import time

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

def setup_fault_patches(nx, ny, length, width, depth, dip_deg, strike_deg):
    """Discretize fault plane into rectangular patches."""
    dip_rad = np.deg2rad(dip_deg)
    strike_rad = np.deg2rad(strike_deg)

    dx = length / nx
    dy = width / ny

    patches = []
    for j in range(ny):
        for i in range(nx):
            cx = (i + 0.5) * dx - length / 2
            dip_dist = (j + 0.5) * dy
            cy_offset = dip_dist * np.cos(dip_rad)
            cz = depth + dip_dist * np.sin(dip_rad)

            cx_rot = cx * np.cos(strike_rad)
            cy_rot = cx * np.sin(strike_rad) + cy_offset

            patches.append({
                "cx": cx_rot,
                "cy": cy_rot,
                "depth": cz,
                "length": dx,
                "width": dy,
                "dip_rad": dip_rad,
                "i": i,
                "j": j,
            })

    return patches

def create_gt_slip(nx, ny):
    """
    Create heterogeneous slip distribution on fault plane.
    Simulates an asperity (high-slip zone) surrounded by lower slip.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    slip = 3.0 * np.exp(-((X - 0.4)**2 / 0.04 + (Y - 0.5)**2 / 0.06))
    slip += 1.5 * np.exp(-((X - 0.75)**2 / 0.02 + (Y - 0.3)**2 / 0.03))
    slip += 0.3 * np.exp(-((X - 0.5)**2 / 0.2 + (Y - 0.5)**2 / 0.2))
    slip = np.maximum(slip, 0.0)

    return slip

def generate_observations(n_obs, fault_length, fault_width, seed):
    """Generate observation station positions around the fault."""
    np.random.seed(seed)
    obs_x = np.random.uniform(-fault_length, fault_length, n_obs)
    obs_y = np.random.uniform(-fault_width * 0.5, fault_width * 2.0, n_obs)
    return np.column_stack([obs_x, obs_y])

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

def load_and_preprocess_data(nx_fault, ny_fault, n_obs, fault_length, fault_width,
                              fault_depth, fault_dip, fault_strike, noise_level, seed, poisson):
    """
    Load and preprocess data for finite fault inversion.
    
    Creates:
    - Ground truth slip distribution
    - Observation station coordinates
    - Fault patch parameters
    - Green's function matrix
    - Synthetic observed displacements with noise
    
    Returns:
        dict containing all preprocessed data needed for inversion
    """
    print("[1] Setting up fault plane ...")
    patches = setup_fault_patches(nx_fault, ny_fault, fault_length, fault_width,
                                  fault_depth, fault_dip, fault_strike)
    n_patches = len(patches)
    print(f"    {nx_fault}×{ny_fault} = {n_patches} patches")
    print(f"    Fault: {fault_length}×{fault_width} km, dip={fault_dip}°")

    print("[2] Creating ground truth slip distribution ...")
    gt_slip = create_gt_slip(nx_fault, ny_fault)
    gt_slip_vec = gt_slip.ravel()
    print(f"    Max slip: {gt_slip.max():.2f} m")
    print(f"    Mean slip: {gt_slip.mean():.2f} m")

    print("[3] Generating observation stations ...")
    obs_coords = generate_observations(n_obs, fault_length, fault_width, seed)
    print(f"    {n_obs} stations")

    print("[4] Building Green's function matrix ...")
    t0 = time.time()
    G = build_greens_matrix(obs_coords, patches, poisson)
    t_green = time.time() - t0
    print(f"    G shape: {G.shape}, built in {t_green:.1f}s")

    print("[5] Computing synthetic displacements ...")
    d_true = G @ gt_slip_vec
    np.random.seed(seed + 1)
    noise = noise_level * np.abs(d_true).max() * np.random.randn(len(d_true))
    d_obs = d_true + noise
    print(f"    Max displacement: {np.abs(d_true).max():.4f} m")

    data = {
        "gt_slip": gt_slip,
        "gt_slip_vec": gt_slip_vec,
        "obs_coords": obs_coords,
        "patches": patches,
        "G": G,
        "d_obs": d_obs,
        "d_true": d_true,
        "nx_fault": nx_fault,
        "ny_fault": ny_fault,
        "fault_length": fault_length,
        "fault_width": fault_width,
    }

    return data
