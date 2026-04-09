import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_shepp_logan(size):
    """
    Generate the Shepp-Logan phantom.
    Standard CT test image with ellipses of varying attenuation.
    """
    img = np.zeros((size, size))
    Y, X = np.mgrid[:size, :size]
    X = (X - size / 2) / (size / 2)
    Y = (Y - size / 2) / (size / 2)

    ellipses = [
        (1.0,   0.69,  0.92,  0,      0,       0),
        (-0.8,  0.6624, 0.8740, 0,     -0.0184, 0),
        (-0.2,  0.1100, 0.3100, 0.22,  0,       -18),
        (-0.2,  0.1600, 0.4100, -0.22, 0,       18),
        (0.1,   0.2100, 0.2500, 0,     0.35,    0),
        (0.1,   0.0460, 0.0460, 0,     0.1,     0),
        (0.1,   0.0460, 0.0460, 0,     -0.1,    0),
        (0.1,   0.0460, 0.0230, -0.08, -0.605,  0),
        (0.1,   0.0230, 0.0230, 0,     -0.606,  0),
        (0.1,   0.0230, 0.0460, 0.06,  -0.605,  0),
    ]

    for A, a, b, x0, y0, phi_deg in ellipses:
        phi = np.radians(phi_deg)
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        x_rot = (X - x0) * cos_p + (Y - y0) * sin_p
        y_rot = -(X - x0) * sin_p + (Y - y0) * cos_p

        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        img[mask] += A

    img = np.clip(img, 0, None)
    return img
