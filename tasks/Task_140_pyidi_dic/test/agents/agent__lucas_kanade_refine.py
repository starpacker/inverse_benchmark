import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import map_coordinates

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
