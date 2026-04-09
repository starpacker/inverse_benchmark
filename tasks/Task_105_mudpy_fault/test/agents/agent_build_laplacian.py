import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def build_laplacian(nx, ny):
    """Build 2D Laplacian smoothing matrix for fault patches."""
    n = nx * ny
    L = np.zeros((n, n))

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            count = 0

            if i > 0:
                L[idx, idx - 1] = -1.0
                count += 1
            if i < nx - 1:
                L[idx, idx + 1] = -1.0
                count += 1
            if j > 0:
                L[idx, idx - nx] = -1.0
                count += 1
            if j < ny - 1:
                L[idx, idx + nx] = -1.0
                count += 1

            L[idx, idx] = float(count)

    return L
