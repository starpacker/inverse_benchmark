import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_scene(n_range, n_cross, scene_size):
    """
    Create a scene with point targets + extended targets.
    Returns 2D reflectivity map σ(x,y).
    """
    sigma = np.zeros((n_cross, n_range))
    cx, cy = n_cross // 2, n_range // 2

    # Point targets at various positions
    targets = [
        (cx, cy, 1.0),           # Centre
        (cx - 15, cy - 20, 0.8),
        (cx + 10, cy + 15, 0.6),
        (cx - 20, cy + 25, 0.7),
        (cx + 25, cy - 10, 0.9),
        (cx + 5, cy + 30, 0.5),
    ]
    for tx, ty, amp in targets:
        if 0 <= tx < n_cross and 0 <= ty < n_range:
            sigma[tx, ty] = amp

    # Extended target: small rectangular structure
    sigma[cx-3:cx+3, cy+8:cy+12] = 0.4

    # L-shaped structure
    sigma[cx+10:cx+15, cy-15:cy-10] = 0.5
    sigma[cx+10:cx+12, cy-15:cy-5] = 0.5

    return sigma
