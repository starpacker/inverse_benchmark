import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import map_coordinates

from scipy.signal import fftconvolve

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def _interpolate_image(image, y_coords, x_coords):
    """Bilinear interpolation of image at fractional coordinates."""
    return map_coordinates(image, [y_coords, x_coords], order=1,
                           mode='reflect').reshape(y_coords.shape)

def _image_gradients(image):
    """Compute image gradients using central differences."""
    gy = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    return gy, gx

def _zncc_integer_peak(ref_sub, def_region, search_margin):
    """Find integer displacement via ZNCC."""
    n_pix = ref_sub.shape[0] * ref_sub.shape[1]
    ones = np.ones_like(ref_sub)

    ref_zm = ref_sub - ref_sub.mean()
    ref_energy = np.sum(ref_zm**2)
    if ref_energy < 1e-12:
        return 0, 0

    cross = fftconvolve(def_region, ref_zm[::-1, ::-1], mode='valid')
    local_sum = fftconvolve(def_region, ones, mode='valid')
    local_sum2 = fftconvolve(def_region**2, ones, mode='valid')
    local_var = local_sum2 / n_pix - (local_sum / n_pix)**2
    local_var = np.maximum(local_var, 0.0)
    local_energy = local_var * n_pix
    denom = np.sqrt(ref_energy * local_energy)
    denom[denom < 1e-12] = 1e-12
    ncc_map = cross / denom

    peak = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    int_dy = int(peak[0]) - search_margin
    int_dx = int(peak[1]) - search_margin
    return int_dy, int_dx

def _lucas_kanade_refine(ref_image, def_image, cy, cx, half,
                          init_dy, init_dx, n_iter=20, tol=1e-4):
    """Iterative Lucas-Kanade refinement of displacement for a single subset."""
    # Reference subset coordinates
    yy, xx = np.meshgrid(
        np.arange(cy - half, cy + half, dtype=np.float64),
        np.arange(cx - half, cx + half, dtype=np.float64),
        indexing='ij'
    )
    ref_vals = ref_image[cy - half:cy + half, cx - half:cx + half].ravel()

    # Precompute gradients of deformed image
    gy_full, gx_full = _image_gradients(def_image)

    dy_curr = float(init_dy)
    dx_curr = float(init_dx)

    for iteration in range(n_iter):
        # Sample deformed image at current displacement estimate
        sample_y = yy + dy_curr
        sample_x = xx + dx_curr
        def_vals = _interpolate_image(def_image, sample_y, sample_x).ravel()

        # Gradient of deformed image at current position
        gy_vals = _interpolate_image(gy_full, sample_y, sample_x).ravel()
        gx_vals = _interpolate_image(gx_full, sample_y, sample_x).ravel()

        # Residual
        residual = ref_vals - def_vals

        # Build normal equations
        A = np.zeros((2, 2))
        b = np.zeros(2)
        A[0, 0] = np.sum(gy_vals * gy_vals)
        A[0, 1] = np.sum(gy_vals * gx_vals)
        A[1, 0] = A[0, 1]
        A[1, 1] = np.sum(gx_vals * gx_vals)
        b[0] = np.sum(gy_vals * residual)
        b[1] = np.sum(gx_vals * residual)

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            break

        # Solve 2x2 system
        ddy = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
        ddx = (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det

        dy_curr += ddy
        dx_curr += ddx

        if abs(ddy) < tol and abs(ddx) < tol:
            break

    return dy_curr, dx_curr

def run_inversion(ref_image, images, subset_size=48, step=24, 
                  search_margin=6, lk_iterations=30):
    """
    Run DIC inversion using ZNCC (integer) + Lucas-Kanade (sub-pixel).
    
    Args:
        ref_image: Reference speckle image (height, width)
        images: Deformed image sequence (n_frames, height, width)
        subset_size: Size of DIC subset
        step: Grid step size
        search_margin: Search margin for ZNCC
        lk_iterations: Number of Lucas-Kanade iterations
    
    Returns:
        all_dx_recon: Recovered x-displacement fields (n_frames, ny, nx)
        all_dy_recon: Recovered y-displacement fields (n_frames, ny, nx)
        grid_ys: Y coordinates of grid points
        grid_xs: X coordinates of grid points
    """
    h, w = ref_image.shape
    half = subset_size // 2
    margin = half + search_margin
    n_frames = len(images)

    ys = np.arange(margin, h - margin, step)
    xs = np.arange(margin, w - margin, step)
    ny, nx = len(ys), len(xs)

    all_dx_recon = []
    all_dy_recon = []

    for t in range(n_frames):
        def_image = images[t]
        dx_field = np.zeros((ny, nx))
        dy_field = np.zeros((ny, nx))

        for i, cy in enumerate(ys):
            for j, cx in enumerate(xs):
                ref_sub = ref_image[cy - half:cy + half,
                                    cx - half:cx + half]

                # Search region
                y0 = cy - half - search_margin
                y1 = cy + half + search_margin
                x0 = cx - half - search_margin
                x1 = cx + half + search_margin
                def_region = def_image[y0:y1, x0:x1]

                # Stage 1: integer peak via ZNCC
                int_dy, int_dx = _zncc_integer_peak(ref_sub, def_region,
                                                     search_margin)

                # Stage 2: Lucas-Kanade refinement
                dy_refined, dx_refined = _lucas_kanade_refine(
                    ref_image, def_image, cy, cx, half,
                    init_dy=float(int_dy), init_dx=float(int_dx),
                    n_iter=lk_iterations
                )

                dy_field[i, j] = dy_refined
                dx_field[i, j] = dx_refined

        all_dx_recon.append(dx_field)
        all_dy_recon.append(dy_field)

        # Print per-frame progress
        print(f"       Frame {t:2d}: completed DIC inversion")

    all_dx_recon = np.array(all_dx_recon)
    all_dy_recon = np.array(all_dy_recon)

    return all_dx_recon, all_dy_recon, ys, xs
