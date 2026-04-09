import numpy as np

def _gen_ellipse(x_grid, y_grid, x0, y0, a, b, gray_level, theta=0):
    """Generates a single ellipse mask scaled by gray_level."""
    c = np.cos(theta)
    s = np.sin(theta)
    x_rot = (x_grid - x0) * c + (y_grid - y0) * s
    y_rot = -(x_grid - x0) * s + (y_grid - y0) * c
    mask = (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) <= 1.0
    return mask * gray_level

def _gen_shepp_logan(num_rows, num_cols):
    """Generates the Shepp-Logan phantom."""
    sl_paras = [
        {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': -0.98},
        {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': -0.02},
        {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.01}
    ]
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    x_grid, y_grid = np.meshgrid(axis_x, axis_y)
    image = np.zeros_like(x_grid)
    for el in sl_paras:
        image += _gen_ellipse(x_grid, y_grid, el['x0'], el['y0'], el['a'], el['b'], 
                              el['gray_level'], el['theta'] / 180.0 * np.pi)
    return image
