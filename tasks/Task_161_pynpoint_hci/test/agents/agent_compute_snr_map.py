import numpy as np

import matplotlib

matplotlib.use("Agg")

def compute_snr_map(image):
    """Pixel-wise SNR map using annular noise estimation."""
    ny, nx = image.shape
    cy, cx_img = ny // 2, nx // 2
    yy, xx = np.mgrid[:ny, :nx]
    r_map = np.sqrt((yy - cy) ** 2 + (xx - cx_img) ** 2)

    snr_map = np.zeros_like(image)
    max_r = int(r_map.max()) + 1
    for r in range(3, max_r):
        annulus = (r_map >= r - 1.5) & (r_map < r + 1.5)
        vals = image[annulus]
        if len(vals) > 10:
            std = np.std(vals)
            mean = np.mean(vals)
            if std > 1e-10:
                ring = (r_map >= r - 0.5) & (r_map < r + 0.5)
                snr_map[ring] = (image[ring] - mean) / std
    return snr_map
