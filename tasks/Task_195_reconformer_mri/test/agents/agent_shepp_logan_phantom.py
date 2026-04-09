import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def shepp_logan_phantom(N=256):
    """Generate modified Shepp-Logan phantom of size NxN."""
    ellipses = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0),
        (0.1, 0.046, 0.046, 0.0, 0.1, 0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0),
        (0.1, 0.023, 0.023, 0.0, -0.605, 0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0),
    ]
    img = np.zeros((N, N), dtype=np.float64)
    yc, xc = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    for val, a, b, x0, y0, ang in ellipses:
        th = np.radians(ang)
        ct, st = np.cos(th), np.sin(th)
        xr = ct * (xc - x0) + st * (yc - y0)
        yr = -st * (xc - x0) + ct * (yc - y0)
        img[(xr / a)**2 + (yr / b)**2 <= 1.0] += val
    return img
