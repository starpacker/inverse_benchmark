import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_sky_model(nx=128, ny=128, rng=None):
    """Create a synthetic sky model with point sources and extended emission."""
    if rng is None:
        rng = np.random.default_rng(42)

    sky = np.zeros((ny, nx), dtype=np.float64)

    # 3 point sources at different positions/fluxes
    point_sources = [
        (64, 64, 1.0),    # center, bright
        (40, 80, 0.5),    # offset, medium
        (90, 45, 0.3),    # offset, dim
    ]
    for y, x, flux in point_sources:
        sky[y, x] = flux

    # 1 extended Gaussian source (simulating a galaxy)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = 75, 55
    sigma_x, sigma_y = 5.0, 3.5
    angle = np.pi / 6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = xx - cx
    dy = yy - cy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    gauss = 0.4 * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    sky += gauss

    return sky
