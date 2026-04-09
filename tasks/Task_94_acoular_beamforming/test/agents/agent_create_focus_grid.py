import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_focus_grid(grid_span, grid_res, z_focus):
    """2D focus grid at distance z_focus."""
    coords = np.linspace(-grid_span / 2, grid_span / 2, grid_res)
    gx, gy = np.meshgrid(coords, coords)
    grid_points = np.column_stack([gx.ravel(), gy.ravel(),
                                   np.full(grid_res**2, z_focus)])
    return grid_points, coords
