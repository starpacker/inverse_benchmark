import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def create_gt_object(n):
    """Create a simple 2D binary object (cross + dots pattern)."""
    obj = np.zeros((n, n), dtype=np.float64)
    c = n // 2

    # Central cross
    obj[c-8:c+8, c-2:c+2] = 1.0
    obj[c-2:c+2, c-8:c+8] = 1.0

    # Corner dots
    for dy, dx in [(-15, -15), (-15, 15), (15, -15), (15, 15)]:
        yy, xx = np.ogrid[:n, :n]
        r = np.sqrt((yy - (c + dy))**2 + (xx - (c + dx))**2)
        obj[r < 4] = 1.0

    # Small rectangle
    obj[c+8:c+14, c-12:c-6] = 0.7

    # Triangle-like shape
    for i in range(8):
        obj[c-18+i, c+6:c+6+2*i+1] = 0.8

    return obj
