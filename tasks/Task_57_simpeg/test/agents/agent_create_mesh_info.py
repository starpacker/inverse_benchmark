import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

N_CELLS_X = 20

N_CELLS_Y = 20

N_CELLS_Z = 10

CELL_SIZE_X = 50.0

CELL_SIZE_Y = 50.0

CELL_SIZE_Z = 25.0

def create_mesh_info():
    """Create mesh information dictionary."""
    hx = np.ones(N_CELLS_X) * CELL_SIZE_X
    hy = np.ones(N_CELLS_Y) * CELL_SIZE_Y
    hz = np.ones(N_CELLS_Z) * CELL_SIZE_Z
    
    # Origin so that surface is at z=0, mesh extends downward
    origin = np.array([
        -N_CELLS_X * CELL_SIZE_X / 2,
        -N_CELLS_Y * CELL_SIZE_Y / 2,
        -N_CELLS_Z * CELL_SIZE_Z,
    ])
    
    # Compute cell centers
    x_centers = origin[0] + np.cumsum(hx) - hx / 2
    y_centers = origin[1] + np.cumsum(hy) - hy / 2
    z_centers = origin[2] + np.cumsum(hz) - hz / 2
    
    # Create 3D grid of cell centers (Fortran order for consistency)
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    cell_centers = np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]
    
    mesh_info = {
        'hx': hx,
        'hy': hy,
        'hz': hz,
        'origin': origin,
        'shape_cells': (N_CELLS_X, N_CELLS_Y, N_CELLS_Z),
        'n_cells': N_CELLS_X * N_CELLS_Y * N_CELLS_Z,
        'cell_centers': cell_centers,
        'cell_volumes': CELL_SIZE_X * CELL_SIZE_Y * CELL_SIZE_Z,
    }
    return mesh_info
