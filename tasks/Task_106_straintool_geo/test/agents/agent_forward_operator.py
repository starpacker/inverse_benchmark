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

def forward_operator(strain_field, grid_x, grid_y, query_points):
    """
    Forward operator: compute velocities from strain rate tensor field.
    
    Physics:
      v(x) = ε × (x - x_0) + ω × (x - x_0) + t
      
    Integrates the strain field to compute displacement velocities at query points:
      vx(x,y) = ∫_{x0}^x εxx(x',y) dx' + ∫_{y0}^y εxy(x0,y') dy'
      vy(x,y) = ∫_{y0}^y εyy(x,y') dy' + ∫_{x0}^x εxy(x',y0) dx'
    
    Args:
        strain_field: tuple of (exx, exy, eyy) arrays on the grid
        grid_x: x coordinates of the grid
        grid_y: y coordinates of the grid
        query_points: Nx2 array of (x, y) positions to compute velocities
        
    Returns:
        vx, vy: velocity components at query points
    """
    exx, exy, eyy = strain_field
    
    n_fine = 100
    fine_x = np.linspace(grid_x.min(), grid_x.max(), n_fine)
    fine_y = np.linspace(grid_y.min(), grid_y.max(), n_fine)
    
    interp_exx = RegularGridInterpolator((grid_y, grid_x), exx,
                                         bounds_error=False, fill_value=None)
    interp_exy = RegularGridInterpolator((grid_y, grid_x), exy,
                                         bounds_error=False, fill_value=None)
    interp_eyy = RegularGridInterpolator((grid_y, grid_x), eyy,
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
    
    # Interpolate velocities to query points
    interp_vx = RegularGridInterpolator((fine_y, fine_x), vx_grid,
                                         bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((fine_y, fine_x), vy_grid,
                                         bounds_error=False, fill_value=0.0)
    
    vx = np.array([float(interp_vx((pt[1], pt[0]))) for pt in query_points])
    vy = np.array([float(interp_vy((pt[1], pt[0]))) for pt in query_points])
    
    return vx, vy
