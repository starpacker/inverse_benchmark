"""
Ghost Imaging — Compressed-Sensing Single-Pixel Reconstruction
================================================================
Task #69: Reconstruct an image from single-pixel bucket detector
          measurements using compressive ghost imaging.

Inverse Problem:
    Given M bucket measurements b_i = <φ_i, x> + n_i, where φ_i are
    random speckle illumination patterns and x is the unknown image,
    recover x from M < N measurements (compressed sensing).

Forward Model:
    b = Φ x + n
    Φ is (M × N) measurement matrix (random speckle patterns),
    x is (N × 1) vectorised image, b is (M × 1) bucket signals.

Inverse Solver:
    1) Traditional correlation ghost imaging: x̂ = Σ (b_i - <b>) · φ_i
    2) ISTA (Iterative Shrinkage-Thresholding Algorithm) for CS
    3) FISTA (Fast ISTA) with TV or wavelet sparsity

Repo: https://github.com/cbasedlf/ghost_imaging_demo
Paper: Shapiro (2008), Phys. Rev. A; Katz et al. (2009), APL.

Usage: /data/yjh/spectro_env/bin/python ghost_imaging_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 32                # Image size (32×32)
N_PIXELS = IMG_SIZE ** 2     # Total pixels
COMPRESSION_RATIO = 0.8     # M/N measurement ratio
N_MEASUREMENTS = int(COMPRESSION_RATIO * N_PIXELS)
NOISE_SNR_DB = 50            # Bucket detector SNR
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_test_image(size):
    """
    Create a test image with geometric features:
    - Letters/shapes for visual assessment
    - Smooth gradients for testing reconstruction fidelity
    """
    img = np.zeros((size, size))
    cx, cy = size // 2, size // 2

    # Circle
    Y, X = np.mgrid[:size, :size]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    img[r < size // 4] = 0.8

    # Square
    sq_s = size // 8
    img[cx-sq_s:cx+sq_s, size//4-sq_s:size//4+sq_s] = 0.6

    # Triangle
    for i in range(size // 6):
        j_start = 3 * size // 4 - i
        j_end = 3 * size // 4 + i
        row = cy - size // 6 + i
        if 0 <= row < size and 0 <= j_start and j_end < size:
            img[row, j_start:j_end] = 0.7

    # Small bright features
    img[size // 6, size // 6] = 1.0
    img[5 * size // 6, size // 6] = 1.0
    img[size // 6, 5 * size // 6] = 1.0
    img[5 * size // 6, 5 * size // 6] = 1.0

    # Smooth
    img = gaussian_filter(img, sigma=1)
    img = img / max(img.max(), 1e-12)

    return img


def generate_speckle_patterns(n_measurements, n_pixels, rng):
    """
    Generate random binary speckle illumination patterns.
    Models the spatial light modulator (SLM) patterns.
    """
    # Gaussian random patterns (better RIP properties)
    Phi = rng.standard_normal((n_measurements, n_pixels)) / np.sqrt(n_measurements)
    return Phi


def forward_operator(Phi, x, snr_db, rng):
    """
    Bucket detector measurement: b = Φ @ x + noise.
    """
    b_clean = Phi @ x
    sig_power = np.mean(b_clean**2)
    noise_power = sig_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(len(b_clean))
    b_noisy = b_clean + noise
    return b_clean, b_noisy


# ─── Inverse Solver: Correlation Ghost Imaging ────────────────────
def correlation_gi(Phi, b):
    """
    Traditional correlation ghost imaging:
    x̂ = (1/M) Σ_i (b_i - <b>) · (φ_i - <φ>)
    """
    M = len(b)
    b_mean = b.mean()
    Phi_mean = Phi.mean(axis=0)
    x_rec = np.zeros(Phi.shape[1])
    for i in range(M):
        x_rec += (b[i] - b_mean) * (Phi[i] - Phi_mean)
    x_rec /= M
    return x_rec


# ─── Inverse Solver: ISTA ─────────────────────────────────────────
def dct2d_basis_transform(x, size, inverse=False):
    """2D DCT sparsifying transform."""
    X = x.reshape(size, size)
    if inverse:
        return idct(idct(X, axis=0, norm='ortho'), axis=1, norm='ortho').ravel()
    else:
        return dct(dct(X, axis=0, norm='ortho'), axis=1, norm='ortho').ravel()


def soft_threshold(x, tau):
    """Soft thresholding (proximal operator for L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def ista_cs(Phi, b, img_size, n_iter=300, lam=0.01):
    """
    ISTA (Iterative Shrinkage-Thresholding) for compressed sensing.
    Minimise: (1/2)||Φx - b||² + λ||Ψx||₁
    where Ψ is DCT sparsifying transform.
    """
    M, N = Phi.shape

    # Lipschitz constant
    L = np.linalg.norm(Phi.T @ Phi, ord=2)
    step = 1.0 / L

    x = np.zeros(N)
    print(f"  ISTA: {n_iter} iterations, λ={lam:.4f}, step={step:.6f}")

    for it in range(n_iter):
        # Gradient step
        residual = Phi @ x - b
        grad = Phi.T @ residual
        z = x - step * grad

        # Sparsity in DCT domain
        z_dct = dct2d_basis_transform(z, img_size)
        z_dct = soft_threshold(z_dct, lam * step)
        x = dct2d_basis_transform(z_dct, img_size, inverse=True)

        # Non-negativity
        x = np.maximum(x, 0)

        if (it + 1) % 50 == 0:
            obj = 0.5 * np.linalg.norm(residual)**2 + lam * np.sum(np.abs(z_dct))
            print(f"    iter {it+1:4d}: obj={obj:.4f}")

    return x


def fista_cs(Phi, b, img_size, n_iter=500, lam=0.01):
    """
    FISTA (Fast ISTA) with Nesterov acceleration.
    """
    M, N = Phi.shape
    L = np.linalg.norm(Phi.T @ Phi, ord=2)
    step = 1.0 / L

    x = np.zeros(N)
    y = x.copy()
    t = 1.0

    print(f"  FISTA: {n_iter} iterations, λ={lam:.4f}")

    for it in range(n_iter):
        # Gradient on y
        residual = Phi @ y - b
        grad = Phi.T @ residual
        z = y - step * grad

        # Proximal (DCT domain soft threshold)
        z_dct = dct2d_basis_transform(z, img_size)
        z_dct = soft_threshold(z_dct, lam * step)
        x_new = dct2d_basis_transform(z_dct, img_size, inverse=True)
        x_new = np.maximum(x_new, 0)

        # Nesterov momentum
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

        if (it + 1) % 100 == 0:
            obj = 0.5 * np.linalg.norm(Phi @ x - b)**2 + lam * np.sum(np.abs(z_dct))
            print(f"    iter {it+1:4d}: obj={obj:.4f}")

    return x


# ─── Total Variation Denoising ─────────────────────────────────────
def tv_denoise(img, weight=0.1, n_iter=50):
    """Simple isotropic TV denoising (Chambolle's projection)."""
    u = img.copy()
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    tau = 0.25

    for _ in range(n_iter):
        # Gradient of u
        gx = np.diff(u, axis=0, append=u[-1:, :])
        gy = np.diff(u, axis=1, append=u[:, -1:])

        # Update dual
        px_new = px + tau * gx
        py_new = py + tau * gy
        norm = np.sqrt(px_new**2 + py_new**2)
        norm = np.maximum(norm / weight, 1)
        px = px_new / norm
        py = py_new / norm

        # Divergence
        div_x = np.diff(px, axis=0, prepend=np.zeros((1, u.shape[1])))
        div_y = np.diff(py, axis=1, prepend=np.zeros((u.shape[0], 1)))
        u = img + weight * (div_x + div_y)

    return u


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt, rec):
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(gt, rec_corr, rec_fista, metrics, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')

    rec_corr_n = rec_corr / max(rec_corr.max(), 1e-12)
    axes[1].imshow(rec_corr_n, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Correlation GI')

    rec_fista_n = rec_fista / max(rec_fista.max(), 1e-12)
    axes[2].imshow(rec_fista_n, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('FISTA CS')

    err = np.abs(gt - rec_fista_n)
    axes[3].imshow(err, cmap='hot', vmin=0)
    axes[3].set_title('|Error|')

    for ax in axes:
        ax.axis('off')

    fig.suptitle(
        f"Ghost Imaging — Compressive Single-Pixel Reconstruction\n"
        f"M/N={COMPRESSION_RATIO:.0%} | PSNR={metrics['PSNR']:.1f} dB | "
        f"SSIM={metrics['SSIM']:.4f} | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Ghost Imaging — Compressive Single-Pixel Reconstruction")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation")
    img_gt = generate_test_image(IMG_SIZE)
    x_gt = img_gt.ravel()
    print(f"  Image: {IMG_SIZE}×{IMG_SIZE} = {N_PIXELS} pixels")
    print(f"  Measurements: {N_MEASUREMENTS} (ratio={COMPRESSION_RATIO:.0%})")

    # Stage 2: Forward — Measurement
    print("\n[STAGE 2] Forward — Speckle Illumination + Bucket Detection")
    Phi = generate_speckle_patterns(N_MEASUREMENTS, N_PIXELS, rng)
    b_clean, b_noisy = forward_operator(Phi, x_gt, NOISE_SNR_DB, rng)
    print(f"  Measurement matrix Φ: {Phi.shape}")
    print(f"  Bucket signal range: [{b_noisy.min():.3f}, {b_noisy.max():.3f}]")

    # Stage 3a: Correlation GI
    print("\n[STAGE 3a] Inverse — Correlation Ghost Imaging")
    x_corr = correlation_gi(Phi, b_noisy)
    x_corr = np.maximum(x_corr, 0)
    img_corr = x_corr.reshape(IMG_SIZE, IMG_SIZE)
    m_corr = compute_metrics(img_gt, img_corr)
    print(f"  Correlation GI: CC={m_corr['CC']:.4f}, PSNR={m_corr['PSNR']:.1f}")

    # Stage 3b: FISTA CS
    print("\n[STAGE 3b] Inverse — FISTA Compressed Sensing")
    x_fista = fista_cs(Phi, b_noisy, IMG_SIZE, n_iter=1000, lam=0.001)
    img_fista = x_fista.reshape(IMG_SIZE, IMG_SIZE)
    img_fista = np.clip(img_fista, 0, 1)

    # TV post-processing
    img_fista = tv_denoise(img_fista, weight=0.03, n_iter=50)
    m_fista = compute_metrics(img_gt, img_fista)
    print(f"  FISTA: CC={m_fista['CC']:.4f}, PSNR={m_fista['PSNR']:.1f}")

    # Choose best
    if m_fista['CC'] >= m_corr['CC']:
        img_rec = img_fista
        metrics = m_fista
        method = "FISTA"
    else:
        img_rec = img_corr
        metrics = m_corr
        method = "Correlation"
    print(f"\n  → Using {method} (higher CC)")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), img_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), img_gt)

    visualize_results(img_gt, img_corr, img_fista, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
