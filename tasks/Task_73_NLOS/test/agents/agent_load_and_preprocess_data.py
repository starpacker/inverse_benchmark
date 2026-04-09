import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

WALL_SIZE = 2.0

SCENE_DEPTH_MIN = 0.5

SCENE_DEPTH_MAX = 1.5

def load_and_preprocess_data(n_scan, n_depth, n_time, dt, noise_snr_db, seed):
    """
    Generate hidden scene geometry and simulate NLOS transient measurements.
    
    Parameters:
        n_scan: int - Number of scan points per dimension
        n_depth: int - Number of depth slices in hidden scene
        n_time: int - Number of time bins
        dt: float - Time bin width [s]
        noise_snr_db: float - Signal-to-noise ratio in dB
        seed: int - Random seed for reproducibility
    
    Returns:
        data_dict: dict containing:
            - 'rho_gt': Ground truth hidden scene albedo (nx, ny, nz)
            - 'transient_clean': Clean transient measurements (nx, ny, nt)
            - 'transient_noisy': Noisy transient measurements (nx, ny, nt)
            - 'x_wall': Wall x-coordinates (nx,)
            - 'y_wall': Wall y-coordinates (ny,)
            - 'z_scene': Scene depth coordinates (nz,)
            - 'dt': Time step
    """
    rng = np.random.default_rng(seed)
    
    # Generate hidden scene
    x = np.linspace(-WALL_SIZE / 2, WALL_SIZE / 2, n_scan)
    y = np.linspace(-WALL_SIZE / 2, WALL_SIZE / 2, n_scan)
    z = np.linspace(SCENE_DEPTH_MIN, SCENE_DEPTH_MAX, n_depth)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    rho = np.zeros((n_scan, n_scan, n_depth))
    
    # Flat panel at z ≈ 1.0m
    iz = np.argmin(np.abs(z - 1.0))
    rho[n_scan//4:3*n_scan//4, n_scan//4:3*n_scan//4, iz] = 0.8
    
    # Sphere
    cx, cy, cz = 0.0, 0.3, 0.8
    r_sphere = 0.15
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
    mask = dist < r_sphere
    rho[mask] = np.maximum(rho[mask], 0.6)
    
    # Small bright point
    ix_p = np.argmin(np.abs(x - (-0.3)))
    iy_p = np.argmin(np.abs(y - (-0.2)))
    iz_p = np.argmin(np.abs(z - 1.2))
    rho[ix_p, iy_p, iz_p] = 1.0
    
    # Forward NLOS measurement
    nx, ny = len(x), len(y)
    nz = len(z)
    transient = np.zeros((nx, ny, n_time))
    
    # Get non-zero voxel coordinates
    nz_idx = np.argwhere(rho > 1e-10)
    
    if len(nz_idx) > 0:
        rho_nz = rho[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]
        xs_nz = x[nz_idx[:, 0]]
        ys_nz = y[nz_idx[:, 1]]
        zs_nz = z[nz_idx[:, 2]]
        
        for ix_w in range(nx):
            xw = x[ix_w]
            for iy_w in range(ny):
                yw = y[iy_w]
                dist_arr = np.sqrt((xw - xs_nz)**2 + (yw - ys_nz)**2 + zs_nz**2)
                t_round = 2 * dist_arr / C
                it = (t_round / dt).astype(int)
                valid = (it >= 0) & (it < n_time)
                if not np.any(valid):
                    continue
                cos_theta = zs_nz[valid] / np.maximum(dist_arr[valid], 1e-10)
                weight = cos_theta**2 / np.maximum(dist_arr[valid]**4, 1e-20)
                contributions = rho_nz[valid] * weight
                np.add.at(transient[ix_w, iy_w], it[valid], contributions)
    
    transient_clean = transient.copy()
    
    # Add noise
    sig_power = np.mean(transient**2)
    if sig_power > 0:
        noise_power = sig_power / (10**(noise_snr_db / 10))
        noise = np.sqrt(noise_power) * rng.standard_normal(transient.shape)
        transient_noisy = transient + noise
    else:
        transient_noisy = transient.copy()
    
    data_dict = {
        'rho_gt': rho,
        'transient_clean': transient_clean,
        'transient_noisy': transient_noisy,
        'x_wall': x,
        'y_wall': y,
        'z_scene': z,
        'dt': dt
    }
    
    return data_dict
