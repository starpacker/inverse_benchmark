"""
promptmr_mri - Multi-Contrast MRI Reconstruction
=================================================
Task: Reconstruct multi-contrast MRI from differently undersampled k-space
Repo: https://github.com/hellopipu/PromptMR
Usage: /data/yjh/promptmr_mri_env/bin/python promptmr_mri_code.py

Approach:
- Generate T1-weighted and T2-weighted Shepp-Logan phantoms (256x256)
- Undersample k-space with Cartesian masks (T1: 4x, T2: 8x)
- Reconstruct using FISTA with Total Variation regularization
- Evaluate PSNR/SSIM per contrast and averaged
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. Multi-contrast phantom generation
# ============================================================

def _shepp_logan_ellipses_contrast(contrast='T1'):
    """
    Modified Shepp-Logan ellipses with tissue-dependent intensities.
    Each ellipse: (intensity, a, b, x0, y0, theta_deg)
    Tissues: outer skull, brain WM, GM regions, CSF ventricles, lesions.
    """
    if contrast == 'T1':
        # T1-weighted: bright WM, moderate GM, dark CSF
        return [
            (0.80, 0.6900, 0.9200, 0.0000, 0.0000, 0),    # outer skull
            (-0.60, 0.6624, 0.8740, 0.0000, -0.0184, 0),   # brain interior
            (0.50, 0.1100, 0.3100, 0.2200, 0.0000, -18),   # right WM (bright)
            (0.50, 0.1600, 0.4100, -0.2200, 0.0000, 18),   # left WM (bright)
            (0.30, 0.2100, 0.2500, 0.0000, 0.3500, 0),     # GM region top
            (0.25, 0.0460, 0.0460, 0.0000, 0.1000, 0),     # small GM
            (0.25, 0.0460, 0.0460, 0.0000, -0.1000, 0),    # small GM
            (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0),   # CSF (dark)
            (0.10, 0.0230, 0.0230, 0.0000, -0.6050, 0),    # CSF (dark)
            (0.15, 0.0230, 0.0460, 0.0600, -0.6050, 0),    # CSF (dark)
        ]
    else:  # T2
        # T2-weighted: dark WM, moderate GM, bright CSF
        return [
            (0.70, 0.6900, 0.9200, 0.0000, 0.0000, 0),    # outer skull
            (-0.50, 0.6624, 0.8740, 0.0000, -0.0184, 0),   # brain interior
            (0.20, 0.1100, 0.3100, 0.2200, 0.0000, -18),   # right WM (dark)
            (0.20, 0.1600, 0.4100, -0.2200, 0.0000, 18),   # left WM (dark)
            (0.45, 0.2100, 0.2500, 0.0000, 0.3500, 0),     # GM region top (brighter)
            (0.40, 0.0460, 0.0460, 0.0000, 0.1000, 0),     # small GM
            (0.40, 0.0460, 0.0460, 0.0000, -0.1000, 0),    # small GM
            (0.60, 0.0460, 0.0230, -0.0800, -0.6050, 0),   # CSF (bright)
            (0.60, 0.0230, 0.0230, 0.0000, -0.6050, 0),    # CSF (bright)
            (0.55, 0.0230, 0.0460, 0.0600, -0.6050, 0),    # CSF (bright)
        ]


def generate_phantom(N, contrast='T1'):
    """Generate an NxN phantom image for a given contrast."""
    img = np.zeros((N, N), dtype=np.float64)
    ellipses = _shepp_logan_ellipses_contrast(contrast)

    y_coords, x_coords = np.mgrid[-1:1:N*1j, -1:1:N*1j]

    for (intensity, a, b, x0, y0, theta_deg) in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotate coordinates
        xr = cos_t * (x_coords - x0) + sin_t * (y_coords - y0)
        yr = -sin_t * (x_coords - x0) + cos_t * (y_coords - y0)

        # Ellipse equation
        mask = (xr / a)**2 + (yr / b)**2 <= 1.0
        img[mask] += intensity

    # Clip and normalize to [0, 1]
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return img


# ============================================================
# 2. Forward operator: FFT + undersampling
# ============================================================

def fft2c(img):
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def ifft2c(kspace):
    """Centered 2D IFFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


def create_cartesian_mask(N, acceleration, acs_fraction=0.08, seed=42):
    """
    Create a 1D Cartesian undersampling mask (same for all columns).
    Keeps center ACS lines and randomly selects remaining lines.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros(N, dtype=bool)

    # ACS region (center lines)
    acs_lines = int(N * acs_fraction)
    center = N // 2
    acs_start = center - acs_lines // 2
    acs_end = acs_start + acs_lines
    mask[acs_start:acs_end] = True

    # Number of additional lines to sample
    total_lines = N // acceleration
    remaining = max(0, total_lines - acs_lines)

    # Available non-ACS lines
    non_acs_indices = np.where(~mask)[0]
    if remaining > 0 and len(non_acs_indices) > 0:
        chosen = rng.choice(non_acs_indices, size=min(remaining, len(non_acs_indices)), replace=False)
        mask[chosen] = True

    # Expand to 2D (phase-encode direction = rows)
    mask_2d = np.zeros((N, N), dtype=bool)
    for i in range(N):
        if mask[i]:
            mask_2d[i, :] = True

    return mask_2d


def forward_op(img, mask):
    """Apply forward operator: FFT then mask."""
    kspace = fft2c(img)
    return kspace * mask


def adjoint_op(kspace_masked, mask):
    """Apply adjoint: zero-fill then IFFT."""
    return ifft2c(kspace_masked * mask)


# ============================================================
# 3. TV regularization utilities
# ============================================================

def gradient_2d(img):
    """Compute discrete gradient (finite differences)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy


def divergence_2d(gx, gy):
    """Compute divergence (adjoint of gradient)."""
    dx = np.zeros_like(gx)
    dy = np.zeros_like(gy)
    dx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
    dx[:, 0] = gx[:, 0]
    dx[:, -1] = -gx[:, -2]
    dy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
    dy[0, :] = gy[0, :]
    dy[-1, :] = -gy[-2, :]
    return dx + dy


def tv_prox(img, lam, n_inner=80):
    """
    Proximal operator for isotropic TV using Chambolle's projection algorithm.
    Solves: argmin_x 0.5*||x - img||^2 + lam*TV(x)
    
    Uses the dual formulation from Chambolle (2004).
    """
    if lam <= 0:
        return img.copy()

    px = np.zeros_like(img)
    py = np.zeros_like(img)
    tau = 1.0 / 8.0  # safe step size for 2D gradient (1/(2*dim))

    for _ in range(n_inner):
        # Compute divergence of dual variable
        div_p = divergence_2d(px, py)
        # Gradient of (div_p - img/lam)
        gx, gy = gradient_2d(div_p - img / lam)
        # Update dual variables with projection
        px_new = px + tau * gx
        py_new = py + tau * gy
        # Project onto unit ball
        norm = np.sqrt(px_new**2 + py_new**2)
        norm = np.maximum(norm, 1.0)
        px = px_new / norm
        py = py_new / norm

    return img - lam * divergence_2d(px, py)


# ============================================================
# 4. FISTA solver for CS-MRI reconstruction
# ============================================================

def fista_tv_recon(kspace_masked, mask, lam_tv=0.001, n_iter=300, verbose=True):
    """
    FISTA with TV regularization for MRI reconstruction.
    
    min_x 0.5 * ||F_mask(x) - y||^2 + lam_tv * TV(x)
    
    where F_mask is the masked Fourier operator.
    The Lipschitz constant of the gradient of the data term is 1 
    (since mask is a projection and FFT is unitary).
    """
    # Initial: zero-filled reconstruction
    x = np.real(ifft2c(kspace_masked))
    z = x.copy()
    t = 1.0
    
    # Step size = 1/L, L=1 for masked FFT
    step = 1.0

    for k in range(n_iter):
        x_old = x.copy()

        # Gradient of data fidelity: Re{ F^H (mask * (F*z) - y) }
        fz = fft2c(z)
        residual = mask * fz - kspace_masked
        grad = np.real(ifft2c(residual))

        # Gradient step
        z_step = z - step * grad

        # Proximal step (TV denoising)
        x = tv_prox(z_step, lam_tv * step, n_inner=50)

        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new

        if verbose and (k + 1) % 50 == 0:
            data_fit = np.linalg.norm(fft2c(x) * mask - kspace_masked) / np.linalg.norm(kspace_masked)
            print(f"  Iter {k+1}/{n_iter}, relative residual = {data_fit:.6f}")

    return x


# ============================================================
# 5. Metrics
# ============================================================

def compute_psnr(gt, recon):
    """Compute PSNR."""
    mse = np.mean((gt - recon)**2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10.0 * np.log10(data_range**2 / mse)


def compute_ssim(gt, recon):
    """Compute SSIM."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)


# ============================================================
# 6. Main pipeline
# ============================================================

def main():
    print("=" * 60)
    print("PromptMR-style Multi-Contrast MRI Reconstruction")
    print("=" * 60)

    N = 256
    os.makedirs('results', exist_ok=True)

    # --- Generate phantoms ---
    print("\n[1/4] Generating multi-contrast phantoms...")
    t1_gt = generate_phantom(N, contrast='T1')
    t2_gt = generate_phantom(N, contrast='T2')
    print(f"  T1 phantom: shape={t1_gt.shape}, range=[{t1_gt.min():.3f}, {t1_gt.max():.3f}]")
    print(f"  T2 phantom: shape={t2_gt.shape}, range=[{t2_gt.min():.3f}, {t2_gt.max():.3f}]")

    # --- Create undersampling masks ---
    print("\n[2/4] Creating undersampling masks...")
    mask_t1 = create_cartesian_mask(N, acceleration=4, acs_fraction=0.08, seed=42)
    mask_t2 = create_cartesian_mask(N, acceleration=8, acs_fraction=0.08, seed=123)
    print(f"  T1 mask: {mask_t1.sum()/(N*N)*100:.1f}% sampled (4x acceleration)")
    print(f"  T2 mask: {mask_t2.sum()/(N*N)*100:.1f}% sampled (8x acceleration)")

    # --- Forward operation: generate undersampled k-space ---
    print("\n[3/4] Forward operation + reconstruction...")
    kspace_t1 = forward_op(t1_gt, mask_t1)
    kspace_t2 = forward_op(t2_gt, mask_t2)

    # Zero-filled reconstructions (baseline)
    zf_t1 = np.abs(ifft2c(kspace_t1))
    zf_t2 = np.abs(ifft2c(kspace_t2))

    # --- FISTA-TV reconstruction for each contrast ---
    print("\n  Reconstructing T1 (4x)...")
    recon_t1 = fista_tv_recon(kspace_t1, mask_t1, lam_tv=0.0008, n_iter=250, verbose=True)

    print("\n  Reconstructing T2 (8x)...")
    recon_t2 = fista_tv_recon(kspace_t2, mask_t2, lam_tv=0.001, n_iter=250, verbose=True)

    # --- Evaluate ---
    print("\n[4/4] Evaluating metrics...")
    psnr_zf_t1 = compute_psnr(t1_gt, zf_t1)
    psnr_zf_t2 = compute_psnr(t2_gt, zf_t2)
    ssim_zf_t1 = compute_ssim(t1_gt, zf_t1)
    ssim_zf_t2 = compute_ssim(t2_gt, zf_t2)

    psnr_t1 = compute_psnr(t1_gt, recon_t1)
    psnr_t2 = compute_psnr(t2_gt, recon_t2)
    ssim_t1 = compute_ssim(t1_gt, recon_t1)
    ssim_t2 = compute_ssim(t2_gt, recon_t2)

    psnr_avg = (psnr_t1 + psnr_t2) / 2.0
    ssim_avg = (ssim_t1 + ssim_t2) / 2.0

    print(f"\n  Zero-filled baselines:")
    print(f"    T1: PSNR={psnr_zf_t1:.2f} dB, SSIM={ssim_zf_t1:.4f}")
    print(f"    T2: PSNR={psnr_zf_t2:.2f} dB, SSIM={ssim_zf_t2:.4f}")
    print(f"\n  CS-TV Reconstruction:")
    print(f"    T1 (4x): PSNR={psnr_t1:.2f} dB, SSIM={ssim_t1:.4f}")
    print(f"    T2 (8x): PSNR={psnr_t2:.2f} dB, SSIM={ssim_t2:.4f}")
    print(f"    Average: PSNR={psnr_avg:.2f} dB, SSIM={ssim_avg:.4f}")

    # --- Save metrics ---
    metrics = {
        "task": "promptmr_mri",
        "method": "FISTA-TV CS-MRI Reconstruction",
        "psnr_t1": round(psnr_t1, 2),
        "ssim_t1": round(ssim_t1, 4),
        "psnr_t2": round(psnr_t2, 2),
        "ssim_t2": round(ssim_t2, 4),
        "psnr_avg": round(psnr_avg, 2),
        "ssim_avg": round(ssim_avg, 4),
        "psnr_zf_t1": round(psnr_zf_t1, 2),
        "psnr_zf_t2": round(psnr_zf_t2, 2),
        "t1_acceleration": 4,
        "t2_acceleration": 8,
        "image_size": N,
        "fista_iterations": 250,
    }
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to results/metrics.json")

    # --- Save arrays ---
    # Stack contrasts: shape (2, N, N), index 0=T1, 1=T2
    gt_stack = np.stack([t1_gt, t2_gt], axis=0)
    recon_stack = np.stack([recon_t1, recon_t2], axis=0)
    np.save('results/ground_truth.npy', gt_stack)
    np.save('results/reconstruction.npy', recon_stack)
    print(f"  Arrays saved: ground_truth.npy {gt_stack.shape}, reconstruction.npy {recon_stack.shape}")

    # --- Visualization ---
    print("\n  Generating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Error scaling
    vmax_err = 0.15

    # Row 1: T1
    axes[0, 0].imshow(t1_gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('T1 Ground Truth', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(zf_t1, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'T1 Zero-filled (4x)\nPSNR={psnr_zf_t1:.1f}dB', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(recon_t1, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'T1 CS-TV Recon\nPSNR={psnr_t1:.1f}dB, SSIM={ssim_t1:.3f}', fontsize=12)
    axes[0, 2].axis('off')

    err_t1 = np.abs(t1_gt - recon_t1)
    im1 = axes[0, 3].imshow(err_t1, cmap='hot', vmin=0, vmax=vmax_err)
    axes[0, 3].set_title('T1 Error (×5)', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)

    # Row 2: T2
    axes[1, 0].imshow(t2_gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('T2 Ground Truth', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(zf_t2, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'T2 Zero-filled (8x)\nPSNR={psnr_zf_t2:.1f}dB', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(recon_t2, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'T2 CS-TV Recon\nPSNR={psnr_t2:.1f}dB, SSIM={ssim_t2:.3f}', fontsize=12)
    axes[1, 2].axis('off')

    err_t2 = np.abs(t2_gt - recon_t2)
    im2 = axes[1, 3].imshow(err_t2, cmap='hot', vmin=0, vmax=vmax_err)
    axes[1, 3].set_title('T2 Error (×5)', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)

    fig.suptitle(
        f'Multi-Contrast MRI Reconstruction (PromptMR-style)\n'
        f'T1(4x): PSNR={psnr_t1:.2f}dB/SSIM={ssim_t1:.4f}  |  '
        f'T2(8x): PSNR={psnr_t2:.2f}dB/SSIM={ssim_t2:.4f}  |  '
        f'Avg: PSNR={psnr_avg:.2f}dB/SSIM={ssim_avg:.4f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('results/reconstruction_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Visualization saved to results/reconstruction_result.png")

    print("\n" + "=" * 60)
    print(f"DONE — Average PSNR: {psnr_avg:.2f} dB, SSIM: {ssim_avg:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
