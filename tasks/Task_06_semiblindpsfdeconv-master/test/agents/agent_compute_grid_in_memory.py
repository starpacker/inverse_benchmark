import logging

import numpy as np

from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def compute_grid_in_memory(psf_map_shape, input_image_shape):
    """
    Computes interpolation grid coefficients.
    psf_map_shape: tuple (rows, cols)
    input_image_shape: tuple (H, W)
    """
    grid_z1 = []
    grid_x, grid_y = np.mgrid[0:input_image_shape[0], 0:input_image_shape[1]]
    xmax = np.linspace(0, input_image_shape[0], psf_map_shape[0])
    ymax = np.linspace(0, input_image_shape[1], psf_map_shape[1])

    total_patches = psf_map_shape[0] * psf_map_shape[1]
    
    for i in range(total_patches):
        points = []
        values = []
        for x in xmax:
            for y in ymax:
                points.append(np.asarray([x, y]))
                values.append(0.0)

        values[i] = 1.0
        points = np.asarray(points)
        values = np.asarray(values)
        grid_z1.append(griddata(points, values, (grid_x, grid_y), method='linear', rescale=True))
    
    return grid_z1
