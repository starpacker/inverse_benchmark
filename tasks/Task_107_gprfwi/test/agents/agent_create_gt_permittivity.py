import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.ndimage import gaussian_filter

def create_gt_permittivity(nz, nx):
    """
    Create a 2D permittivity model with horizontal layers and anomalies.
    Shape: (nz, nx).  Values represent relative permittivity ε_r.
    """
    eps = np.ones((nz, nx)) * 4.0       # background ε_r = 4 (dry sand)

    # Layer 1: top soil (ε_r = 6) from z=30 to z=60
    eps[30:60, :] = 6.0

    # Layer 2: wet clay (ε_r = 15) from z=80 to z=120
    eps[80:120, :] = 15.0

    # Layer 3: bedrock (ε_r = 8) from z=150 onward
    eps[150:, :] = 8.0

    # Anomaly 1: buried pipe (ε_r = 1, air) — circle at (z=45, x=25), r=5
    zz, xx = np.ogrid[:nz, :nx]
    mask1 = (zz - 45)**2 + (xx - 25)**2 < 5**2
    eps[mask1] = 1.0

    # Anomaly 2: water pocket (ε_r = 80) — ellipse at (z=100, x=55)
    mask2 = ((zz - 100) / 6)**2 + ((xx - 55) / 8)**2 < 1
    eps[mask2] = 40.0

    # Anomaly 3: metallic object (ε_r = 30) — small rect at (z=140, x=40)
    eps[137:143, 37:43] = 30.0

    # Smooth slightly to avoid unrealistically sharp transitions
    eps = gaussian_filter(eps, sigma=1.0)
    return eps
