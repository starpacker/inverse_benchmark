import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def shepp_logan_phantom(n=128):
    """Generate a Shepp-Logan phantom of size n x n."""
    ellipses = [
        (1.0,   0.6900, 0.9200,  0.0000,  0.0000,   0),
        (-0.8,  0.6624, 0.8740,  0.0000, -0.0184,   0),
        (-0.2,  0.1100, 0.3100,  0.2200,  0.0000, -18),
        (-0.2,  0.1600, 0.4100, -0.2200,  0.0000,  18),
        (0.1,   0.2100, 0.2500,  0.0000,  0.3500,   0),
        (0.1,   0.0460, 0.0460,  0.0000,  0.1000,   0),
        (0.1,   0.0460, 0.0460,  0.0000, -0.1000,   0),
        (0.1,   0.0460, 0.0230, -0.0800, -0.6050,   0),
        (0.1,   0.0230, 0.0230,  0.0000, -0.6060,   0),
        (0.1,   0.0230, 0.0460,  0.0600, -0.6050,   0),
    ]

    phantom = np.zeros((n, n), dtype=np.float64)
    y_coords, x_coords = np.mgrid[-1:1:n*1j, -1:1:n*1j]

    for intensity, a, b, x0, y0, theta_deg in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = cos_t * (x_coords - x0) + sin_t * (y_coords - y0)
        y_rot = -sin_t * (x_coords - x0) + cos_t * (y_coords - y0)
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
        phantom[mask] += intensity

    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-12)
    return phantom
