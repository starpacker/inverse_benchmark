import matplotlib

matplotlib.use('Agg')

import numpy as np

def shepp_logan_phantom(N=128):
    """Generate a Shepp-Logan phantom of size NxN."""
    ellipses = [
        (2.0, 0.6900, 0.9200, 0.0000, 0.0000, 0),
        (-0.98, 0.6624, 0.8740, 0.0000, -0.0184, 0),
        (-0.02, 0.1100, 0.3100, 0.2200, 0.0000, -18),
        (-0.02, 0.1600, 0.4100, -0.2200, 0.0000, 18),
        (0.01, 0.2100, 0.2500, 0.0000, 0.3500, 0),
        (0.01, 0.0460, 0.0460, 0.0000, 0.1000, 0),
        (0.01, 0.0460, 0.0460, 0.0000, -0.1000, 0),
        (0.01, 0.0460, 0.0230, -0.0800, -0.6050, 0),
        (0.01, 0.0230, 0.0230, 0.0000, -0.6060, 0),
        (0.01, 0.0230, 0.0460, 0.0600, -0.6050, 0),
    ]

    img = np.zeros((N, N), dtype=np.float64)
    ygrid, xgrid = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]

    for intensity, a, b, x0, y0, theta_deg in ellipses:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = cos_t * (xgrid - x0) + sin_t * (ygrid - y0)
        yr = -sin_t * (xgrid - x0) + cos_t * (ygrid - y0)
        region = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        img[region] += intensity

    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img
