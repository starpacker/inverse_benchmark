import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.ndimage import gaussian_filter, zoom

from skimage.restoration import denoise_tv_chambolle

def run_inversion(
    y_obs,
    target_size,
    scale_factor=4,
    aa_sigma=1.0,
    tv_weight_stage1=0.08,
    tv_weight_stage2=0.04,
    num_datafid_iters=20,
    datafid_step=0.3
):
    """
    DDRM-inspired SVD-based super-resolution restoration.

    This implements the DDRM algorithm spirit using classical tools:

    Stage 1 — SVD Pseudo-Inverse Initialisation:
      Bicubic upsampling provides the SVD pseudo-inverse estimate.

    Stage 2 — Prior Regularisation (TV Denoising):
      TV denoising serves as the prior, analogous to the DDPM score network.

    Stage 3 — Spectral Data-Fidelity Correction:
      Gradient descent on ||y - A(x)||^2 with the correct adjoint A^T.

    Stage 4 — Final TV Polish:
      A lighter TV pass removes residual artifacts.

    Parameters
    ----------
    y_obs : ndarray, shape (LR_SIZE, LR_SIZE)
        Noisy low-resolution observation.
    target_size : int
        Target HR image size.
    scale_factor : int
        Upsampling factor.
    aa_sigma : float
        Sigma for Gaussian blur in forward model.
    tv_weight_stage1 : float
        TV weight for initial denoising.
    tv_weight_stage2 : float
        TV weight for final polish.
    num_datafid_iters : int
        Number of data-fidelity gradient correction steps.
    datafid_step : float
        Step size for data-fidelity gradient descent.

    Returns
    -------
    x : ndarray, shape (target_size, target_size)
        Restored high-resolution image.
    """
    
    def blur_image(img, sigma):
        """Apply Gaussian blur."""
        return gaussian_filter(img, sigma=sigma)
    
    def downsample_block_avg(img, scale):
        """Downsample by block averaging."""
        h, w = img.shape
        return img.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))
    
    def upsample_adjoint(img_lr, scale):
        """Adjoint of block-average downsample: pixel replication scaled by 1/s^2."""
        return np.repeat(np.repeat(img_lr, scale, axis=0),
                         scale, axis=1) / (scale**2)
    
    def forward_A(x, scale, sigma):
        """Apply degradation: A(x) = Downsample(Blur(x))."""
        return downsample_block_avg(blur_image(x, sigma), scale)
    
    def adjoint_AT(y, scale, sigma):
        """Apply adjoint: A^T(y) = Blur^T(Upsample^T(y))."""
        return blur_image(upsample_adjoint(y, scale), sigma)
    
    print(f"\n[DDRM] === Starting SVD-Based Diffusion Restoration ===")
    scale = target_size // y_obs.shape[0]

    # ── Stage 1: SVD Pseudo-Inverse (Bicubic Upsampling) ──
    print(f"[DDRM] Stage 1: SVD pseudo-inverse initialisation (bicubic {scale}x)")
    x = np.clip(zoom(y_obs, scale, order=3), 0, 1)
    print(f"[DDRM]   Bicubic init range: [{x.min():.4f}, {x.max():.4f}]")

    # ── Stage 2: Prior Regularisation via TV (Diffusion Prior Proxy) ──
    print(f"[DDRM] Stage 2: TV regularisation (weight={tv_weight_stage1})")
    print(f"[DDRM]   In DDRM, the DDPM score network regularises the solution.")
    print(f"[DDRM]   Here TV denoising serves as the prior, removing noise while")
    print(f"[DDRM]   preserving edges — projecting onto the natural image manifold.")
    x = denoise_tv_chambolle(x, weight=tv_weight_stage1)
    x = np.clip(x, 0, 1)
    print(f"[DDRM]   Post-TV range: [{x.min():.4f}, {x.max():.4f}]")

    # ── Stage 3: Data-Fidelity Gradient Correction (SVD Spectral Update) ──
    print(f"[DDRM] Stage 3: Data-fidelity correction ({num_datafid_iters} iters)")
    print(f"[DDRM]   In DDRM, this corresponds to updating the SVD spectral")
    print(f"[DDRM]   coefficients s_i to satisfy y = U_y Sigma V^T x.")
    print(f"[DDRM]   Here: gradient descent on 0.5*||y - A(x)||^2 with step={datafid_step}")

    for i in range(num_datafid_iters):
        # Forward: compute A(x) and residual
        Ax = forward_A(x, scale_factor, aa_sigma)
        residual = Ax - y_obs

        # Adjoint gradient: A^T(A(x) - y)
        grad = adjoint_AT(residual, scale_factor, aa_sigma)

        # Gradient descent step
        x = x - datafid_step * grad
        x = np.clip(x, 0, 1)

        if (i + 1) % 5 == 0:
            cost = 0.5 * np.sum(residual**2)
            print(f"[DDRM]   Iter {i+1}/{num_datafid_iters}: "
                  f"data_cost = {cost:.6f}")

    # ── Stage 4: Final TV Polish ──
    print(f"[DDRM] Stage 4: Final TV polish (weight={tv_weight_stage2})")
    x = denoise_tv_chambolle(x, weight=tv_weight_stage2)
    x = np.clip(x, 0, 1)
    print(f"[DDRM]   Final range: [{x.min():.4f}, {x.max():.4f}]")

    print(f"[DDRM] === Restoration Complete ===")
    return x
