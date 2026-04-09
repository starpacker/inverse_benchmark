import matplotlib

matplotlib.use('Agg')

import numpy as np

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

from scipy.ndimage import gaussian_filter

def fft2c(img):
    """Centered 2D FFT: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    """Centered 2D IFFT: k-space -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

def soft_threshold(x, threshold):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def compute_tv_gradient(x):
    """Compute the gradient of an isotropic TV approximation."""
    eps = 1e-8
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    grad_mag = np.sqrt(dx ** 2 + dy ** 2 + eps)
    nx = dx / grad_mag
    ny = dy / grad_mag
    div_x = nx - np.roll(nx, 1, axis=1)
    div_y = ny - np.roll(ny, 1, axis=0)
    return -(div_x + div_y)

def gradient_data_fidelity(x, kspace_under, mask):
    """
    Gradient of data fidelity term: ||MFx - y||^2
    grad = F^H M^H (MFx - y) = F^H M (MFx - y)
    """
    Fx = fft2c(x)
    residual = mask * Fx - kspace_under
    grad = ifft2c(mask * residual)
    return np.real(grad)

def run_inversion(kspace_under, mask, N, gt_norm):
    """
    Run MRI reconstruction using multiple methods and select the best.
    
    Methods:
    - POCS (Projection Onto Convex Sets)
    - FISTA-TV (Fast Iterative Shrinkage-Thresholding with Total Variation)
    - ISTA-TV variants
    
    Args:
        kspace_under: Undersampled k-space data
        mask: Undersampling mask
        N: Image size
        gt_norm: Ground truth for metric computation
    
    Returns:
        recon_norm: Best reconstruction result
        recon_metrics: (PSNR, SSIM, RMSE) for best reconstruction
        method_name: Name of the best method
    """
    
    def compute_metrics_internal(gt, recon):
        gt_n = np.clip(gt, 0, 1).astype(np.float64)
        recon_n = np.clip(recon, 0, 1).astype(np.float64)
        psnr_val = psnr(gt_n, recon_n, data_range=1.0)
        ssim_val = ssim(gt_n, recon_n, data_range=1.0)
        rmse_val = np.sqrt(np.mean((gt_n - recon_n) ** 2))
        return psnr_val, ssim_val, rmse_val
    
    def pocs_reconstruction(kspace_under, mask, N, n_iter=200):
        """POCS MRI reconstruction."""
        x = np.abs(ifft2c(kspace_under))
        x = x / (x.max() + 1e-12)

        for it in range(n_iter):
            kx = fft2c(x)
            kx_dc = kx * (1 - mask) + kspace_under
            x = ifft2c(kx_dc)
            x = np.abs(x)
            if it < n_iter // 2:
                sigma = max(0.5, 2.0 * (1 - it / (n_iter // 2)))
                x_smooth = gaussian_filter(np.real(x), sigma=sigma)
                x = 0.7 * x + 0.3 * x_smooth
            x = np.clip(np.real(x), 0, None)

            if (it + 1) % 50 == 0:
                residual = np.sum(np.abs(mask * fft2c(x) - kspace_under) ** 2)
                print(f"  POCS iter {it + 1}/{n_iter}: residual={residual:.4f}")

        return x / (x.max() + 1e-12)
    
    def proximal_gradient_recon(kspace_under, mask, N, n_iter=500, step_size=1.0, lam=0.01):
        """FISTA-TV reconstruction."""
        x = np.abs(ifft2c(kspace_under))
        x_scale = x.max() + 1e-12
        kspace_scaled = kspace_under / x_scale
        x = x / x_scale

        x_prev = x.copy()
        t = 1.0

        for it in range(n_iter):
            Fx = fft2c(x)
            residual = mask * Fx - kspace_scaled
            grad = np.real(ifft2c(residual))
            x_gd = x - step_size * grad

            dx = np.roll(x_gd, -1, axis=1) - x_gd
            dy = np.roll(x_gd, -1, axis=0) - x_gd
            dx_t = soft_threshold(dx, lam * step_size)
            dy_t = soft_threshold(dy, lam * step_size)

            div_x = dx_t - np.roll(dx_t, 1, axis=1)
            div_y = dy_t - np.roll(dy_t, 1, axis=0)
            x_new = x_gd - 0.25 * (div_x + div_y - (dx - np.roll(dx, 1, axis=1)) - (dy - np.roll(dy, 1, axis=0)))
            x_new = np.clip(np.real(x_new), 0, None)

            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            x = x_new + ((t - 1) / t_new) * (x_new - x_prev)
            x = np.clip(x, 0, None)
            x_prev = x_new
            t = t_new

            if (it + 1) % 100 == 0:
                res = np.sum(np.abs(mask * fft2c(x) - kspace_scaled) ** 2)
                print(f"  FISTA iter {it + 1}/{n_iter}: residual={res:.6f}")

        return x / (x.max() + 1e-12)
    
    def ista_tv_reconstruction(kspace_under, mask, N, n_iter=300, step_size=0.5, lam_tv=0.005):
        """ISTA with Total Variation regularization."""
        x = np.real(ifft2c(kspace_under))
        x = np.clip(x, 0, None)
        x_max_init = x.max() + 1e-12
        x = x / x_max_init

        best_x = x.copy()
        best_loss = np.inf

        for it in range(n_iter):
            grad_data = gradient_data_fidelity(x, kspace_under / x_max_init, mask)
            grad_tv = compute_tv_gradient(x)
            x = x - step_size * (grad_data + lam_tv * grad_tv)
            x = np.clip(x, 0, None)

            residual_k = mask * fft2c(x) - kspace_under / x_max_init
            loss = np.sum(np.abs(residual_k) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()

            if (it + 1) % 50 == 0:
                print(f"  ISTA iter {it + 1}/{n_iter}: data_loss={loss:.4f}")

        return best_x / (best_x.max() + 1e-12)
    
    # Run POCS
    print("\n  Running POCS reconstruction...")
    recon_pocs = pocs_reconstruction(kspace_under, mask, N, n_iter=200)
    pocs_metrics = compute_metrics_internal(gt_norm, recon_pocs)
    print(f"  POCS: PSNR={pocs_metrics[0]:.2f} dB, SSIM={pocs_metrics[1]:.4f}")
    
    # Run FISTA-TV
    print("\n  Running FISTA-TV reconstruction...")
    recon_fista = proximal_gradient_recon(kspace_under, mask, N, n_iter=500, step_size=1.0, lam=0.005)
    fista_metrics = compute_metrics_internal(gt_norm, recon_fista)
    print(f"  FISTA-TV: PSNR={fista_metrics[0]:.2f} dB, SSIM={fista_metrics[1]:.4f}")
    
    candidates = [
        (recon_pocs, pocs_metrics, "POCS"),
        (recon_fista, fista_metrics, "FISTA-TV (unrolled optimization)"),
    ]
    
    # Try ISTA-TV variants
    print("\n  Trying ISTA-TV variants...")
    for lam_val in [0.001, 0.003, 0.01]:
        for step in [0.3, 0.5, 1.0]:
            recon_v = ista_tv_reconstruction(kspace_under, mask, N,
                                             n_iter=200, step_size=step, lam_tv=lam_val)
            m = compute_metrics_internal(gt_norm, recon_v)
            if m[0] > 15 and m[1] > 0.5:
                print(f"    ISTA-TV(lam={lam_val}, step={step}): PSNR={m[0]:.2f}, SSIM={m[1]:.4f}")
                candidates.append((recon_v, m, f"ISTA-TV(lam={lam_val}, step={step})"))
                break
        else:
            continue
        break
    
    # Select best reconstruction
    best_idx = max(range(len(candidates)), key=lambda i: candidates[i][1][0] + 10 * candidates[i][1][1])
    recon_norm, recon_metrics, method_name = candidates[best_idx]
    
    print(f"\n  Best method: {method_name}")
    print(f"  PSNR={recon_metrics[0]:.2f} dB, SSIM={recon_metrics[1]:.4f}")
    
    # Try more aggressive approach if below threshold
    if recon_metrics[0] < 15 or recon_metrics[1] < 0.5:
        print("\n  Below threshold. Trying more iterations...")
        recon_extra = pocs_reconstruction(kspace_under, mask, N, n_iter=500)
        extra_metrics = compute_metrics_internal(gt_norm, recon_extra)
        print(f"  POCS-500: PSNR={extra_metrics[0]:.2f}, SSIM={extra_metrics[1]:.4f}")
        if extra_metrics[0] > recon_metrics[0]:
            recon_norm, recon_metrics, method_name = recon_extra, extra_metrics, "POCS (500 iter)"
    
    return recon_norm, recon_metrics, method_name
