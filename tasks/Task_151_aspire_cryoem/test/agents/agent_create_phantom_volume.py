import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_phantom_volume(L):
    """
    Create a 3D phantom volume with multiple Gaussian blobs and an ellipsoidal
    shell, mimicking a simplified protein structure.
    """
    coords = np.mgrid[:L, :L, :L].astype(np.float64)
    center = (L - 1) / 2.0
    coords = coords - center

    vol = np.zeros((L, L, L), dtype=np.float64)

    # Large central ellipsoid (core of protein)
    r_ellipsoid = np.sqrt((coords[0] / (L * 0.30))**2 +
                          (coords[1] / (L * 0.25))**2 +
                          (coords[2] / (L * 0.20))**2)
    vol += 0.8 * np.exp(-0.5 * r_ellipsoid**2)

    # Several smaller Gaussian blobs (subunits / domains)
    blob_params = [
        (0.20, 0.15, 0.10, 0.08, 1.0),
        (-0.15, -0.20, 0.12, 0.07, 0.9),
        (0.10, -0.10, -0.18, 0.06, 1.1),
        (-0.18, 0.12, -0.08, 0.09, 0.7),
        (0.00, 0.22, 0.00, 0.05, 1.2),
        (0.15, 0.00, -0.15, 0.07, 0.8),
    ]

    for ox, oy, oz, sigma, amp in blob_params:
        dx = coords[0] / L - ox
        dy = coords[1] / L - oy
        dz = coords[2] / L - oz
        r2 = (dx**2 + dy**2 + dz**2) / sigma**2
        vol += amp * np.exp(-0.5 * r2)

    # Normalize to [0, 1]
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)

    return vol
