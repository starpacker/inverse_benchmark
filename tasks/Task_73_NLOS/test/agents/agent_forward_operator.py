import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

def forward_operator(rho, x_wall, y_wall, z_scene, n_time, dt):
    """
    Confocal NLOS forward model.
    
    For each scan point x_s = (x_w, y_w, 0):
      τ(x_s, t_k) = Σ_p ρ(p) · δ(t_k - 2||p - x_s||/c) · w(p, x_s)
    
    where w(p, x_s) = cos²θ / ||p-x_s||⁴ accounts for geometric falloff.
    
    Parameters:
        rho: ndarray (nx, ny, nz) - Hidden scene albedo
        x_wall: ndarray (nx,) - Wall x-coordinates
        y_wall: ndarray (ny,) - Wall y-coordinates
        z_scene: ndarray (nz,) - Scene depth coordinates
        n_time: int - Number of time bins
        dt: float - Time bin width [s]
    
    Returns:
        transient: ndarray (nx, ny, n_time) - Predicted transient measurements
    """
    nx, ny = len(x_wall), len(y_wall)
    transient = np.zeros((nx, ny, n_time))
    
    # Get non-zero voxel coordinates
    nz_idx = np.argwhere(rho > 1e-10)
    
    if len(nz_idx) == 0:
        return transient
    
    rho_nz = rho[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]
    xs_nz = x_wall[nz_idx[:, 0]]
    ys_nz = y_wall[nz_idx[:, 1]]
    zs_nz = z_scene[nz_idx[:, 2]]
    
    for ix_w in range(nx):
        xw = x_wall[ix_w]
        for iy_w in range(ny):
            yw = y_wall[iy_w]
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
    
    return transient
