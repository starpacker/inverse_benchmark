"""
pyxu_imaging — Computational Imaging: 2D Image Deconvolution
=============================================================
Task: Image deconvolution (deblurring) using proximal algorithms
Repo: https://github.com/pyxu-org/pyxu
Library: Pyxu v1.2.0 (formerly Pycsou) — modular proximal algorithm framework

Inverse Problem:
    Given: y = H*x + noise  (blurred + noisy observation)
    Find:  x_hat = argmin_x 0.5*||H*x - y||_2^2 + lambda * ||nabla(x)||_1
    
    where H is a convolution operator (Gaussian PSF),
    and ||nabla(x)||_1 is the isotropic Total Variation (TV) regularizer.

Solver: Condat-Vu primal-dual splitting
    f(x) = 0.5*||Hx - y||^2     (differentiable, Lipschitz gradient)
    h(Kx) = lambda * ||Kx||_1   where K = Gradient operator

Usage:
    /data/yjh/pyxu_imaging_env/bin/python pyxu_imaging_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

# Add repo to path (if needed)
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

# Pyxu imports
import pyxu.operator as pxo
import pyxu.opt.solver as pxs
import pyxu.opt.stop as pxst

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Image parameters
IMG_SIZE = 128          # Image dimension (128x128)
BLUR_SIGMA = 2.0        # Gaussian blur sigma (pixels)
NOISE_LEVEL = 0.02      # Additive white Gaussian noise std
SEED = 42

# Solver parameters
LAMBDA_TV = 0.008       # TV regularization weight
MAX_ITER = 600          # Maximum solver iterations


# ═══════════════════════════════════════════════════════════
# 2. Data Generation: Synthetic Test Image
# ═══════════════════════════════════════════════════════════
def create_test_image(size=IMG_SIZE, seed=SEED):
    """
    Create a synthetic piecewise-smooth test image with geometric shapes.
    Ideal for TV-regularized deconvolution.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.float64)

    # Background gradient
    y_coords, x_coords = np.mgrid[0:size, 0:size] / float(size)
    img += 0.1 * y_coords

    # Rectangle
    img[20:50, 30:80] = 0.7

    # Circle
    cy, cx, r = 80, 40, 20
    yy, xx = np.ogrid[:size, :size]
    mask_circle = (yy - cy)**2 + (xx - cx)**2 <= r**2
    img[mask_circle] = 0.9

    # Small bright square
    img[60:75, 85:100] = 1.0

    # Triangle
    for row in range(30, 60):
        col_start = 85 + (row - 30)
        col_end = 115 - (row - 30)
        if col_start < col_end and col_end <= size:
            img[row, col_start:col_end] = 0.5

    # Diagonal stripe
    for i in range(size):
        j_start = max(0, i - 3)
        j_end = min(size, i + 3)
        if 10 <= i <= 110:
            img[i, j_start:j_end] = np.maximum(img[i, j_start:j_end], 0.4)

    # Small dots
    dot_positions = [(15, 15), (15, 110), (110, 15), (110, 110), (64, 64)]
    for dy, dx in dot_positions:
        if 0 <= dy < size and 0 <= dx < size:
            img[max(0,dy-2):min(size,dy+3), max(0,dx-2):min(size,dx+3)] = 0.85

    img = np.clip(img, 0, 1)
    return img


def create_gaussian_kernel(sigma, size=None):
    """Create a 2D Gaussian convolution kernel."""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator & Observation
# ═══════════════════════════════════════════════════════════
def build_forward_operator(img_shape, blur_sigma):
    """
    Build the forward operator H using Pyxu's Convolve operator.
    H: R^(H*W) -> R^(H*W) (same-size convolution via zero-padding)
    """
    kernel = create_gaussian_kernel(blur_sigma)
    
    H = pxo.Convolve(
        arg_shape=img_shape,
        kernel=kernel,
        center=(kernel.shape[0]//2, kernel.shape[1]//2),
        mode="constant",
    )
    
    return H, kernel


def generate_observation(H, x_true_flat, noise_level, seed=SEED):
    """Apply forward model: y = H(x) + noise."""
    rng = np.random.RandomState(seed + 1)
    y_clean = H(x_true_flat)
    noise = rng.normal(0, noise_level, y_clean.shape)
    y_noisy = y_clean + noise
    return y_noisy, y_clean


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: Condat-Vu Primal-Dual with TV
# ═══════════════════════════════════════════════════════════
def reconstruct_tv(H, y_observed, img_shape, lambda_tv, max_iter):
    """
    Solve: min_x f(x) + h(Kx)
    
    f(x) = 0.5 * ||H(x) - y||_2^2    (differentiable data fidelity)
    h(z) = lambda * ||z||_1           (TV regularizer prox)
    K = Gradient                       (finite differences)
    
    Using Condat-Vu primal-dual splitting (pyxu.opt.solver.CV).
    """
    N = img_shape[0] * img_shape[1]
    
    # Data fidelity: f(x) = 0.5 * ||Hx - y||^2
    sl2 = 0.5 * pxo.SquaredL2Norm(dim=N)
    f = sl2.asloss(y_observed) * H
    
    # TV regularizer: h(z) = lambda * ||z||_1, K = Gradient
    grad_op = pxo.Gradient(arg_shape=img_shape)
    h = lambda_tv * pxo.L1Norm(dim=grad_op.codim)
    K = grad_op
    
    print(f"  Forward op H: dim={H.dim}, codim={H.codim}")
    print(f"  Gradient K: dim={K.dim}, codim={K.codim}")
    print(f"  f: 0.5*||Hx-y||^2  (differentiable)")
    print(f"  h: {lambda_tv} * ||z||_1  (proximable)")
    print(f"  K: Gradient (finite differences)")
    
    # Condat-Vu solver
    solver = pxs.CV(f=f, h=h, K=K, show_progress=False)
    stop = pxst.MaxIter(n=max_iter)
    
    # Initial guess: degraded observation
    x0 = y_observed.copy()
    
    print(f"  Running Condat-Vu solver (max_iter={max_iter})...")
    solver.fit(x0=x0, stop_crit=stop)
    
    x_sol = solver.solution()
    print(f"  Solver finished. Solution shape: {x_sol.shape}")
    return x_sol


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_psnr(ref, test, data_range=None):
    ref = ref.astype(np.float64).ravel()
    test = test.astype(np.float64).ravel()
    if data_range is None:
        data_range = ref.max() - ref.min()
    if data_range < 1e-10:
        data_range = 1.0
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-30:
        return 100.0
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(ref, test):
    from skimage.metrics import structural_similarity as ssim
    r = ref.squeeze()
    t = test.squeeze()
    data_range = r.max() - r.min()
    if data_range < 1e-10:
        data_range = 1.0
    return float(ssim(r, t, data_range=data_range))


def compute_rmse(ref, test):
    return float(np.sqrt(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)))


def compute_correlation(ref, test):
    r = ref.flatten().astype(np.float64)
    t = test.flatten().astype(np.float64)
    r_c = r - r.mean()
    t_c = t - t.mean()
    num = np.sum(r_c * t_c)
    den = np.sqrt(np.sum(r_c**2) * np.sum(t_c**2))
    if den < 1e-30:
        return 0.0
    return float(num / den)


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(ground_truth, observation, reconstruction, metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    gt = ground_truth.squeeze()
    obs = observation.squeeze()
    recon = reconstruction.squeeze()
    error = np.abs(gt - recon)

    vmin, vmax = 0, 1

    im0 = axes[0, 0].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('(a) Ground Truth', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(obs, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'(b) Observation (blur σ={BLUR_SIGMA}, noise={NOISE_LEVEL})',
                         fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'(c) Reconstruction (PSNR={metrics["psnr"]:.2f} dB)',
                         fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 0.01))
    axes[1, 0].set_title(f'(d) |Error| (RMSE={metrics["rmse"]:.4f})',
                         fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    mid_row = gt.shape[0] // 2
    axes[1, 1].plot(gt[mid_row, :], 'b-', label='GT', linewidth=2)
    axes[1, 1].plot(obs[mid_row, :], 'g--', label='Observed', linewidth=1, alpha=0.5)
    axes[1, 1].plot(recon[mid_row, :], 'r-', label='Recon', linewidth=1.5)
    axes[1, 1].set_title(f'(e) Profile at row {mid_row}', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    try:
        import pyxu
        pyxu_ver = pyxu.__version__
    except:
        pyxu_ver = "unknown"
    metrics_text = (
        f"Reconstruction Metrics\n"
        f"{'='*30}\n\n"
        f"PSNR:  {metrics['psnr']:.2f} dB\n"
        f"SSIM:  {metrics['ssim']:.4f}\n"
        f"RMSE:  {metrics['rmse']:.6f}\n"
        f"CC:    {metrics['cc']:.4f}\n\n"
        f"{'='*30}\n"
        f"Solver: Condat-Vu (primal-dual)\n"
        f"Library: Pyxu {pyxu_ver}\n"
        f"lambda_TV: {LAMBDA_TV}\n"
        f"Blur sigma: {BLUR_SIGMA}\n"
        f"Noise: {NOISE_LEVEL}\n"
        f"Image: {IMG_SIZE}x{IMG_SIZE}\n"
        f"Max iter: {MAX_ITER}"
    )
    axes[1, 2].text(0.1, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Pyxu Image Deconvolution: TV-Regularized Proximal Algorithm',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization -> {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  pyxu_imaging — 2D Image Deconvolution")
    print("  Solver: Condat-Vu primal-dual splitting (TV)")
    print("  Library: Pyxu (proximal algorithms)")
    print("=" * 60)

    # (a) Create ground truth image
    print("\n[1/6] Creating test image...")
    x_true = create_test_image(IMG_SIZE, SEED)
    print(f"  Image shape: {x_true.shape}, range: [{x_true.min():.3f}, {x_true.max():.3f}]")

    # (b) Build forward operator
    print("\n[2/6] Building forward operator (Gaussian blur)...")
    H, kernel = build_forward_operator(x_true.shape, BLUR_SIGMA)
    print(f"  Kernel shape: {kernel.shape}, sigma={BLUR_SIGMA}")
    print(f"  H: dim={H.dim}, codim={H.codim}")

    # (c) Generate observation
    print("\n[3/6] Generating observation (blur + noise)...")
    x_true_flat = x_true.ravel()
    y_observed, y_clean = generate_observation(H, x_true_flat, NOISE_LEVEL, SEED)
    
    obs_psnr = compute_psnr(x_true_flat, y_observed)
    print(f"  Observation shape: {y_observed.shape}")
    print(f"  Observation PSNR: {obs_psnr:.2f} dB (degraded)")

    # (d) Run reconstruction
    print("\n[4/6] Running reconstruction...")
    recon_flat = reconstruct_tv(H, y_observed, x_true.shape, LAMBDA_TV, MAX_ITER)
    
    # Clip to valid range
    recon_flat = np.clip(recon_flat, 0, 1)
    recon_2d = recon_flat.reshape(x_true.shape)
    print(f"  Reconstruction range: [{recon_2d.min():.4f}, {recon_2d.max():.4f}]")

    # (e) Evaluate
    print("\n[5/6] Evaluating results...")
    metrics = {
        "psnr": float(compute_psnr(x_true, recon_2d)),
        "ssim": float(compute_ssim(x_true, recon_2d)),
        "rmse": float(compute_rmse(x_true, recon_2d)),
        "cc": float(compute_correlation(x_true, recon_2d)),
        "observation_psnr": float(obs_psnr),
        "solver": "Condat-Vu (primal-dual splitting)",
        "regularizer": "Total Variation (anisotropic, L1 of gradient)",
        "lambda_tv": LAMBDA_TV,
        "blur_sigma": BLUR_SIGMA,
        "noise_level": NOISE_LEVEL,
        "max_iterations": MAX_ITER,
        "image_size": IMG_SIZE,
    }
    print(f"  PSNR = {metrics['psnr']:.4f} dB")
    print(f"  SSIM = {metrics['ssim']:.6f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CC   = {metrics['cc']:.4f}")

    # (f) Save outputs
    print("\n[6/6] Saving outputs...")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics -> {metrics_path}")

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), x_true)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_2d)
    np.save("gt_output.npy", x_true)
    np.save("recon_output.npy", recon_2d)
    print(f"  Arrays saved")

    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(x_true, y_observed.reshape(x_true.shape), recon_2d, metrics, vis_path)

    print("\n" + "=" * 60)
    print(f"  PSNR = {metrics['psnr']:.2f} dB | SSIM = {metrics['ssim']:.4f}")
    print("  DONE")
    print("=" * 60)
