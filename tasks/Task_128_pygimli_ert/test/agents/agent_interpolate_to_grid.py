import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.interpolate import griddata

def interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=100, ny=50):
    """Interpolate cell-based values to a regular grid for visualization."""
    cell_centers = np.array([[mesh.cell(i).center().x(),
                              mesh.cell(i).center().y()]
                             for i in range(mesh.cellCount())])

    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x_coords, y_coords)

    grid_values = griddata(cell_centers, np.array(cell_values),
                           (xx, yy), method='linear', fill_value=np.nan)

    return grid_values, x_coords, y_coords
