import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.interpolate import RegularGridInterpolator

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_106_straintool_geo"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def load_and_preprocess_data(n_stations, region_size, grid_n, noise_level, gauss_sigma, seed):
    """
    Load and preprocess data for geodetic strain rate inversion.
    
    Creates:
    - Grid coordinates for strain output
    - Ground truth strain rate tensor field
    - GNSS station positions
    - Synthetic velocity observations
    
    Returns:
        dict containing all necessary data for inversion
    """
    np.random.seed(seed)
    
    # Create grid
    grid_x = np.linspace(20, region_size - 20, grid_n)
    grid_y = np.linspace(20, region_size - 20, grid_n)
    
    # Create ground truth strain field
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Background strain (regional extension)
    exx_bg = 10.0  # nanostrain/yr
    eyy_bg = -5.0
    exy_bg = 2.0
    
    gt_exx = np.full_like(X, exx_bg)
    gt_eyy = np.full_like(X, eyy_bg)
    gt_exy = np.full_like(X, exy_bg)
    
    # Localized shear zone (fault-like feature)
    fault_x = region_size * 0.5
    fault_y = region_size * 0.5
    fault_width = 60.0  # km
    
    # Gaussian shear anomaly centered on fault
    anomaly = np.exp(-((X - fault_x)**2 + (Y - fault_y)**2) / (2 * fault_width**2))
    
    gt_exx += 80.0 * anomaly   # compression across zone
    gt_eyy += -40.0 * anomaly  # extension along zone
    gt_exy += 120.0 * anomaly  # shear across zone
    
    # Secondary deformation zone
    anomaly2 = np.exp(-((X - region_size * 0.2)**2 + (Y - region_size * 0.7)**2)
                      / (2 * 40.0**2))
    gt_exx += -30.0 * anomaly2
    gt_eyy += 50.0 * anomaly2
    gt_exy += -60.0 * anomaly2
    
    # Generate station positions
    stations = np.column_stack([
        np.random.uniform(30, region_size - 30, n_stations),
        np.random.uniform(30, region_size - 30, n_stations),
    ])
    
    # Generate synthetic velocities using forward model
    n_fine = 100
    fine_x = np.linspace(grid_x.min(), grid_x.max(), n_fine)
    fine_y = np.linspace(grid_y.min(), grid_y.max(), n_fine)
    
    interp_exx = RegularGridInterpolator((grid_y, grid_x), gt_exx,
                                         bounds_error=False, fill_value=None)
    interp_exy = RegularGridInterpolator((grid_y, grid_x), gt_exy,
                                         bounds_error=False, fill_value=None)
    interp_eyy = RegularGridInterpolator((grid_y, grid_x), gt_eyy,
                                         bounds_error=False, fill_value=None)
    
    FX, FY = np.meshgrid(fine_x, fine_y)
    pts_fine = np.column_stack([FY.ravel(), FX.ravel()])
    
    exx_fine = interp_exx(pts_fine).reshape(n_fine, n_fine)
    exy_fine = interp_exy(pts_fine).reshape(n_fine, n_fine)
    eyy_fine = interp_eyy(pts_fine).reshape(n_fine, n_fine)
    
    scale = 1e-3
    cx = n_fine // 2
    cy = n_fine // 2
    ddx = fine_x[1] - fine_x[0]
    ddy = fine_y[1] - fine_y[0]
    
    vx_grid = np.zeros((n_fine, n_fine))
    vy_grid = np.zeros((n_fine, n_fine))
    
    # vx(x,y) = ∫_{x0}^x εxx(x',y) dx' + ∫_{y0}^y εxy(x0,y') dy'
    for j in range(n_fine):
        cum = 0.0
        for i in range(cx + 1, n_fine):
            cum += exx_fine[j, i] * ddx * scale
            vx_grid[j, i] = cum
        cum = 0.0
        for i in range(cx - 1, -1, -1):
            cum -= exx_fine[j, i + 1] * ddx * scale
            vx_grid[j, i] = cum
    
    vx_cross = np.zeros(n_fine)
    cum = 0.0
    for j in range(cy + 1, n_fine):
        cum += exy_fine[j, cx] * ddy * scale
        vx_cross[j] = cum
    cum = 0.0
    for j in range(cy - 1, -1, -1):
        cum -= exy_fine[j + 1, cx] * ddy * scale
        vx_cross[j] = cum
    
    for j in range(n_fine):
        vx_grid[j, :] += vx_cross[j]
    
    # vy(x,y) = ∫_{y0}^y εyy(x,y') dy' + ∫_{x0}^x εxy(x',y0) dx'
    for i in range(n_fine):
        cum = 0.0
        for j in range(cy + 1, n_fine):
            cum += eyy_fine[j, i] * ddy * scale
            vy_grid[j, i] = cum
        cum = 0.0
        for j in range(cy - 1, -1, -1):
            cum -= eyy_fine[j + 1, i] * ddy * scale
            vy_grid[j, i] = cum
    
    vy_cross = np.zeros(n_fine)
    cum = 0.0
    for i in range(cx + 1, n_fine):
        cum += exy_fine[cy, i] * ddx * scale
        vy_cross[i] = cum
    cum = 0.0
    for i in range(cx - 1, -1, -1):
        cum -= exy_fine[cy, i + 1] * ddx * scale
        vy_cross[i] = cum
    
    for i in range(n_fine):
        vy_grid[:, i] += vy_cross[i]
    
    # Interpolate velocities to station locations
    interp_vx = RegularGridInterpolator((fine_y, fine_x), vx_grid,
                                         bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((fine_y, fine_x), vy_grid,
                                         bounds_error=False, fill_value=0.0)
    
    vx = np.array([float(interp_vx((s[1], s[0]))) for s in stations])
    vy = np.array([float(interp_vy((s[1], s[0]))) for s in stations])
    
    # Add noise
    vx += noise_level * np.random.randn(n_stations)
    vy += noise_level * np.random.randn(n_stations)
    
    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'gt_exx': gt_exx,
        'gt_exy': gt_exy,
        'gt_eyy': gt_eyy,
        'stations': stations,
        'vx': vx,
        'vy': vy,
        'gauss_sigma': gauss_sigma,
        'region_size': region_size,
    }
