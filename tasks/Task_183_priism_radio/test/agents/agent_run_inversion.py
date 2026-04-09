import numpy as np

import matplotlib

matplotlib.use('Agg')

def adjoint_operator(vis, ui, vi, nx, ny):
    """
    Adjoint model: visibilities → image (dirty image direction).
    Places visibilities on grid and applies inverse FFT.
    """
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    img = np.fft.ifft2(grid).real
    return img

def make_dirty_image(vis, ui, vi, nx, ny):
    """Create the dirty image (adjoint applied to visibilities), normalized."""
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    psf_grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(psf_grid, (vi, ui), 1.0)
    dirty = np.fft.ifft2(grid).real
    psf = np.fft.ifft2(psf_grid).real
    peak_psf = psf.max()
    if peak_psf > 0:
        dirty /= peak_psf
    return dirty

def tsv_value(image):
    """Total Squared Variation: sum of squared differences."""
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    return np.sum(dx ** 2) + np.sum(dy ** 2)

def tsv_gradient(image):
    """Gradient of TSV(I) w.r.t. I."""
    ny, nx = image.shape
    grad = np.zeros_like(image)
    grad[:, :-1] -= 2 * (image[:, 1:] - image[:, :-1])
    grad[:, 1:] += 2 * (image[:, 1:] - image[:, :-1])
    grad[:-1, :] -= 2 * (image[1:, :] - image[:-1, :])
    grad[1:, :] += 2 * (image[1:, :] - image[:-1, :])
    return grad

def soft_threshold(x, threshold):
    """Proximal operator for L1 norm."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def forward_operator(image, ui, vi):
    """
    Forward model: image → visibilities at (u,v) sample points.
    Uses FFT + sampling.
    
    Parameters:
        image: 2D numpy array (ny, nx) - the sky image
        ui: 1D int array - u grid indices
        vi: 1D int array - v grid indices
    
    Returns:
        vis: 1D complex array - sampled visibilities
    """
    ft = np.fft.fft2(image)
    vis = ft[vi, ui]
    return vis

def run_inversion(vis, ui, vi, nx, ny, lambda_l1=2e-4, lambda_tsv=1e-3,
                  step_size=None, max_iter=800, verbose=True):
    """
    ISTA solver for:
        min_{I>=0} ||Phi*I - V||^2 + lambda_l1*||I||_1 + lambda_tsv*TSV(I)

    Parameters:
        vis: complex array — measured visibilities
        ui, vi: int arrays — grid indices of (u,v) samples
        nx, ny: int — image dimensions
        lambda_l1: float — L1 sparsity weight
        lambda_tsv: float — TSV smoothness weight
        step_size: float or None — if None, estimate from operator norm
        max_iter: int — maximum iterations
        verbose: bool — print progress

    Returns:
        image: 2D array — reconstructed image
        history: dict — convergence history
    """
    print("Step 5: Running ISTA (L1+TSV) reconstruction ...")
    
    # Estimate step size from Lipschitz constant of data fidelity gradient
    if step_size is None:
        x = np.random.randn(ny, nx)
        for _ in range(20):
            y = adjoint_operator(forward_operator(x, ui, vi), ui, vi, nx, ny)
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-14:
                break
            x = y / norm_y
        L = norm_y
        step_size = 0.9 / L
        if verbose:
            print(f"Estimated Lipschitz constant L={L:.2e}, step_size={step_size:.2e}")

    # Initialize with dirty image
    image = make_dirty_image(vis, ui, vi, nx, ny)
    image = np.maximum(image, 0.0)

    history = {'cost': [], 'data_fidelity': [], 'l1': [], 'tsv': []}

    for it in range(max_iter):
        # Gradient of data fidelity: Phi^H (Phi*I - V)
        residual = forward_operator(image, ui, vi) - vis
        grad_data = adjoint_operator(residual, ui, vi, nx, ny)

        # TSV gradient
        grad_tsv = tsv_gradient(image)

        # Gradient descent step
        image = image - step_size * (grad_data + lambda_tsv * grad_tsv)

        # L1 proximal (soft thresholding)
        image = soft_threshold(image, step_size * lambda_l1)

        # Non-negativity constraint
        image = np.maximum(image, 0.0)

        # Track convergence
        if it % 50 == 0 or it == max_iter - 1:
            df = 0.5 * np.sum(np.abs(residual) ** 2)
            l1_val = lambda_l1 * np.sum(np.abs(image))
            tsv_val = lambda_tsv * tsv_value(image)
            cost = df + l1_val + tsv_val
            history['cost'].append(cost)
            history['data_fidelity'].append(df)
            history['l1'].append(l1_val)
            history['tsv'].append(tsv_val)
            if verbose and it % 100 == 0:
                print(f"  iter {it:4d}: cost={cost:.4e}  "
                      f"data={df:.4e}  L1={l1_val:.4e}  TSV={tsv_val:.4e}")

    print(f"  Reconstruction range: [{image.min():.4f}, {image.max():.4f}]")
    
    return image, history
