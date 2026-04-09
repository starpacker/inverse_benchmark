import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def create_gt_slip(nx, ny):
    """
    Create heterogeneous slip distribution on fault plane.
    Simulates an asperity (high-slip zone) surrounded by lower slip.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    slip = 3.0 * np.exp(-((X - 0.4)**2 / 0.04 + (Y - 0.5)**2 / 0.06))
    slip += 1.5 * np.exp(-((X - 0.75)**2 / 0.02 + (Y - 0.3)**2 / 0.03))
    slip += 0.3 * np.exp(-((X - 0.5)**2 / 0.2 + (Y - 0.5)**2 / 0.2))
    slip = np.maximum(slip, 0.0)

    return slip
