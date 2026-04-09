import numpy as np

import matplotlib

matplotlib.use("Agg")

def aperture_sum(image, row, col, radius):
    """Sum of pixel values inside a circular aperture."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
    return image[mask].sum(), mask

def compute_snr_at_position(image, row, col, fwhm):
    """
    Detection SNR (aperture photometry, following Mawet et al. 2014).

    Signal = sum in planet aperture − mean of reference aperture sums.
    Noise  = std of reference aperture sums around the annulus.
    """
    ny, nx = image.shape
    cy, cx_img = ny // 2, nx // 2
    sep = np.sqrt((row - cy) ** 2 + (col - cx_img) ** 2)
    ap_r = fwhm / 2.0

    signal, _ = aperture_sum(image, row, col, ap_r)

    # Reference apertures at the same radial separation
    n_ref = max(int(2 * np.pi * sep / (2 * ap_r + 1)), 8)
    ref_sums = []
    for k in range(n_ref):
        theta = 2 * np.pi * k / n_ref
        rr = cy + sep * np.sin(theta)
        cc = cx_img + sep * np.cos(theta)
        if np.sqrt((rr - row) ** 2 + (cc - col) ** 2) < 3 * ap_r:
            continue
        if rr < ap_r or rr >= ny - ap_r or cc < ap_r or cc >= nx - ap_r:
            continue
        s, _ = aperture_sum(image, rr, cc, ap_r)
        ref_sums.append(s)

    if len(ref_sums) < 3:
        return np.abs(signal) / (np.abs(signal) * 0.1 + 1e-10)

    noise_std = np.std(ref_sums)
    mean_bg = np.mean(ref_sums)
    snr = (signal - mean_bg) / (noise_std + 1e-10)
    return snr
