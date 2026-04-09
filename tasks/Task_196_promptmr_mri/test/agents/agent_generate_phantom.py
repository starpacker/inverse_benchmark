import numpy as np

import matplotlib

matplotlib.use('Agg')

def _shepp_logan_ellipses_contrast(contrast='T1'):
    """
    Modified Shepp-Logan ellipses with tissue-dependent intensities.
    Each ellipse: (intensity, a, b, x0, y0, theta_deg)
    Tissues: outer skull, brain WM, GM regions, CSF ventricles, lesions.
    """
    if contrast == 'T1':
        return [
            (0.80, 0.6900, 0.9200, 0.0000, 0.0000, 0),
            (-0.60, 0.6624, 0.8740, 0.0000, -0.0184, 0),
            (0.50, 0.1100, 0.3100, 0.2200, 0.0000, -18),
            (0.50, 0.1600, 0.4100, -0.2200, 0.0000, 18),
            (0.30, 0.2100, 0.2500, 0.0000, 0.3500, 0),
            (0.25, 0.0460, 0.0460, 0.0000, 0.1000, 0),
            (0.25, 0.0460, 0.0460, 0.0000, -0.1000, 0),
            (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0),
            (0.10, 0.0230, 0.0230, 0.0000, -0.6050, 0),
            (0.15, 0.0230, 0.0460, 0.0600, -0.6050, 0),
        ]
    else:
        return [
            (0.70, 0.6900, 0.9200, 0.0000, 0.0000, 0),
            (-0.50, 0.6624, 0.8740, 0.0000, -0.0184, 0),
            (0.20, 0.1100, 0.3100, 0.2200, 0.0000, -18),
            (0.20, 0.1600, 0.4100, -0.2200, 0.0000, 18),
            (0.45, 0.2100, 0.2500, 0.0000, 0.3500, 0),
            (0.40, 0.0460, 0.0460, 0.0000, 0.1000, 0),
            (0.40, 0.0460, 0.0460, 0.0000, -0.1000, 0),
            (0.60, 0.0460, 0.0230, -0.0800, -0.6050, 0),
            (0.60, 0.0230, 0.0230, 0.0000, -0.6050, 0),
            (0.55, 0.0230, 0.0460, 0.0600, -0.6050, 0),
        ]

def generate_phantom(N, contrast='T1'):
    """Generate an NxN phantom image for a given contrast."""
    img = np.zeros((N, N), dtype=np.float64)
    ellipses = _shepp_logan_ellipses_contrast(contrast)

    y_coords, x_coords = np.mgrid[-1:1:N*1j, -1:1:N*1j]

    for (intensity, a, b, x0, y0, theta_deg) in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        xr = cos_t * (x_coords - x0) + sin_t * (y_coords - y0)
        yr = -sin_t * (x_coords - x0) + cos_t * (y_coords - y0)

        mask = (xr / a)**2 + (yr / b)**2 <= 1.0
        img[mask] += intensity

    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return img
