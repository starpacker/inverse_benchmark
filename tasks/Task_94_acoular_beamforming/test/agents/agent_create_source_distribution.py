import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_source_distribution(grid_points, grid_res):
    """3 Gaussian blob sources."""
    sources = [
        {'x': -0.12, 'y': 0.15,  'strength': 1.0},
        {'x': 0.18,  'y': -0.08, 'strength': 0.7},
        {'x': 0.0,   'y': -0.20, 'strength': 0.5},
    ]
    q = np.zeros(grid_res * grid_res)
    sigma = 0.04
    for s in sources:
        r2 = (grid_points[:, 0] - s['x'])**2 + (grid_points[:, 1] - s['y'])**2
        q += s['strength'] * np.exp(-r2 / (2 * sigma**2))
    return q
