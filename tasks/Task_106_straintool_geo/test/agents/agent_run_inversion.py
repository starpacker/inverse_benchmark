import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

import time

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_106_straintool_geo"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def run_inversion(data, min_stations=10):
    """
    Run weighted least-squares strain rate inversion.
    
    At each grid point, estimates the strain rate tensor using Gaussian-weighted
    velocity gradient estimation from nearby GNSS stations.
    
    Fits a linear model for each velocity component:
      vx(x,y) = a0 + a1*(x-px) + a2*(y-py)
      vy(x,y) = b0 + b1*(x-px) + b2*(y-py)
    
    Then: exx = a1/scale, exy = (a2 + b1)/(2*scale), eyy = b2/scale
    
    Args:
        data: dict from load_and_preprocess_data
        min_stations: minimum number of stations for local inversion
        
    Returns:
        dict containing reconstructed strain fields and timing info
    """
    grid_x = data['grid_x']
    grid_y = data['grid_y']
    stations = data['stations']
    vx = data['vx']
    vy = data['vy']
    sigma = data['gauss_sigma']
    
    ny = len(grid_y)
    nx = len(grid_x)
    n = stations.shape[0]
    
    rec_exx = np.zeros((ny, nx))
    rec_exy = np.zeros((ny, nx))
    rec_eyy = np.zeros((ny, nx))
    
    t0 = time.time()
    
    for j in range(ny):
        for i in range(nx):
            px = grid_x[i]
            py = grid_y[j]
            
            # Gaussian weights centered on the query point
            dist = np.sqrt((stations[:, 0] - px)**2 + (stations[:, 1] - py)**2)
            weights = np.exp(-dist**2 / (2 * sigma**2))
            
            # Only use stations with significant weight
            mask = weights > 0.01 * weights.max()
            if np.sum(mask) < min_stations:
                idx = np.argsort(dist)[:min_stations]
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True
            
            w = weights[mask]
            sx = stations[mask, 0]
            sy = stations[mask, 1]
            vx_sel = vx[mask]
            vy_sel = vy[mask]
            
            n_sel = len(w)
            
            # Displacement from query point
            dx = sx - px
            dy = sy - py
            
            # Build design matrix for linear fit: v = a0 + a1*dx + a2*dy
            A = np.column_stack([np.ones(n_sel), dx, dy])
            
            # Weighted least squares
            W = np.diag(w)
            AW = A.T @ W
            
            try:
                reg = 1e-10 * np.eye(3)
                ax = np.linalg.solve(AW @ A + reg, AW @ vx_sel)
                ay = np.linalg.solve(AW @ A + reg, AW @ vy_sel)
            except np.linalg.LinAlgError:
                rec_exx[j, i] = 0.0
                rec_exy[j, i] = 0.0
                rec_eyy[j, i] = 0.0
                continue
            
            # Scale factor
            scale = 1e-3
            
            # Extract strain from velocity gradients
            rec_exx[j, i] = ax[1] / scale
            rec_eyy[j, i] = ay[2] / scale
            rec_exy[j, i] = 0.5 * (ax[2] + ay[1]) / scale
    
    t_inv = time.time() - t0
    
    return {
        'rec_exx': rec_exx,
        'rec_exy': rec_exy,
        'rec_eyy': rec_eyy,
        'inversion_time': t_inv,
    }
