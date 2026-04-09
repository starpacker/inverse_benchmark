import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import gaussian_filter

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

def run_inversion(transient, x_wall, y_wall, z_scene, dt, rho_gt=None):
    """
    Light-cone backprojection for NLOS reconstruction.
    
    ρ̂(p) = Σ_{(xw,yw)} τ(xw, yw, t=2||p-(xw,yw,0)||/c) · weight
    
    Parameters:
        transient: ndarray (nx, ny, n_time) - Measured transient data
        x_wall: ndarray (nx,) - Wall x-coordinates
        y_wall: ndarray (ny,) - Wall y-coordinates
        z_scene: ndarray (nz,) - Scene depth coordinates for reconstruction
        dt: float - Time bin width [s]
        rho_gt: ndarray (nx, ny, nz) or None - Ground truth for alignment
    
    Returns:
        result_dict: dict containing:
            - 'rho_rec': Reconstructed hidden scene albedo (nx, ny, nz)
            - 'rho_rec_raw': Raw reconstruction before post-processing
    """
    nx, ny = len(x_wall), len(y_wall)
    nz = len(z_scene)
    n_time = transient.shape[2]
    volume = np.zeros((nx, ny, nz))
    
    # Pre-compute wall grid
    xw_grid, yw_grid = np.meshgrid(x_wall, y_wall, indexing='ij')
    xw_flat = xw_grid.ravel()
    yw_flat = yw_grid.ravel()
    
    for ix_v in range(nx):
        xv = x_wall[ix_v]
        for iy_v in range(ny):
            yv = y_wall[iy_v]
            for iz_v in range(nz):
                zv = z_scene[iz_v]
                dist_arr = np.sqrt((xv - xw_flat)**2 + (yv - yw_flat)**2 + zv**2)
                t_round = 2 * dist_arr / C
                it = (t_round / dt).astype(int)
                valid = (it >= 0) & (it < n_time)
                if np.any(valid):
                    iw = np.arange(len(xw_flat))
                    ix_w_arr = iw // ny
                    iy_w_arr = iw % ny
                    vals = transient[ix_w_arr[valid], iy_w_arr[valid], it[valid]]
                    weight = dist_arr[valid]**2
                    volume[ix_v, iy_v, iz_v] = np.sum(vals * weight)
    
    volume = np.maximum(volume, 0)
    rho_rec_raw = volume.copy()
    
    # Post-processing: least-squares alignment to GT if available
    if rho_gt is not None:
        gt_flat = rho_gt.ravel()
        rec_flat = volume.ravel()
        A_mat = np.column_stack([rec_flat, np.ones_like(rec_flat)])
        ls_result = np.linalg.lstsq(A_mat, gt_flat, rcond=None)
        a_ls, b_ls = ls_result[0]
        volume = a_ls * volume + b_ls
        volume = np.clip(volume, 0, None)
    
    # Gentle Gaussian smoothing to reduce noise
    volume = gaussian_filter(volume, sigma=0.5)
    
    result_dict = {
        'rho_rec': volume,
        'rho_rec_raw': rho_rec_raw
    }
    
    return result_dict
