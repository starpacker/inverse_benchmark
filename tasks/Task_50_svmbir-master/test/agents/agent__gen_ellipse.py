import numpy as np

def _gen_ellipse(x_grid, y_grid, x0, y0, a, b, gray_level, theta=0):
    """Generates a single ellipse mask scaled by gray_level."""
    c = np.cos(theta)
    s = np.sin(theta)
    x_rot = (x_grid - x0) * c + (y_grid - y0) * s
    y_rot = -(x_grid - x0) * s + (y_grid - y0) * c
    mask = (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) <= 1.0
    return mask * gray_level
