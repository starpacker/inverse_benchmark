"""
TIGRE — Sparse-View CT Reconstruction
=======================================
Task #72: Reconstruct a 2D tomographic image from sparse-view
          projections using Filtered Back-Projection (FBP) and
          iterative (SIRT/CGLS) algorithms.

Inverse Problem:
    Given sinogram data p(θ,s) = R[f](θ,s) (Radon transform of f),
    with sparse angular sampling, recover image f(x,y).

Forward Model:
    Radon transform (line integrals):
    p(θ,s) = ∫∫ f(x,y) δ(x cosθ + y sinθ - s) dx dy

Inverse Solver:
    1) FBP with Ram-Lak (ramp) filter
    2) SIRT (Simultaneous Iterative Reconstruction Technique)
    3) CGLS (Conjugate Gradient Least Squares)
    All with sparse angular sampling.

Repo: https://github.com/CERN/TIGRE
Paper: Biguri et al. (2016), Biomedical Physics & Engineering Express.

Usage: /data/yjh/spectro_env/bin/python TIGRE_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn
from skimage.transform import iradon

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 128              # Image size (128×128)
N_ANGLES_FULL = 180         # Full-view number of angles
N_ANGLES_SPARSE = 60        # Sparse-view number of angles
N_DETECTORS = int(np.ceil(IMG_SIZE * np.sqrt(2)))  # Detector elements
NOISE_SNR_DB = 35           # Poisson-like noise level
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_shepp_logan(size):
    """
    Generate the Shepp-Logan phantom.
    Standard CT test image with ellipses of varying attenuation.
    """
    img = np.zeros((size, size))
    Y, X = np.mgrid[:size, :size]
    X = (X - size / 2) / (size / 2)
    Y = (Y - size / 2) / (size / 2)

    # Ellipse parameters: (A, a, b, x0, y0, phi)
    ellipses = [
        (1.0,   0.69,  0.92,  0,      0,       0),      # Outer skull
        (-0.8,  0.6624, 0.8740, 0,     -0.0184, 0),      # Brain
        (-0.2,  0.1100, 0.3100, 0.22,  0,       -18),    # Left ventricle
        (-0.2,  0.1600, 0.4100, -0.22, 0,       18),     # Right ventricle
        (0.1,   0.2100, 0.2500, 0,     0.35,    0),      # Top tumour
        (0.1,   0.0460, 0.0460, 0,     0.1,     0),      # Small tumour 1
        (0.1,   0.0460, 0.0460, 0,     -0.1,    0),      # Small tumour 2
        (0.1,   0.0460, 0.0230, -0.08, -0.605,  0),      # Bottom left
        (0.1,   0.0230, 0.0230, 0,     -0.606,  0),      # Bottom centre
        (0.1,   0.0230, 0.0460, 0.06,  -0.605,  0),      # Bottom right
    ]

    for A, a, b, x0, y0, phi_deg in ellipses:
        phi = np.radians(phi_deg)
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        x_rot = (X - x0) * cos_p + (Y - y0) * sin_p
        y_rot = -(X - x0) * sin_p + (Y - y0) * cos_p

        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        img[mask] += A

    img = np.clip(img, 0, None)
    return img


# ─── Forward Operator: Radon Transform ────────────────────────────
def radon_transform(image, angles_deg):
    """
    Compute the Radon transform (sinogram) of a 2D image.
    Uses rotation + integration approach.
    """
    n = image.shape[0]
    n_det = int(np.ceil(n * np.sqrt(2)))
    if n_det % 2 == 0:
        n_det += 1
    sinogram = np.zeros((len(angles_deg), n_det))

    # Pad image to n_det × n_det
    pad_total = n_det - n
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    img_padded = np.pad(image, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')

    center = img_padded.shape[0] / 2

    from scipy.ndimage import rotate as ndi_rotate

    for i, angle in enumerate(angles_deg):
        rotated = ndi_rotate(img_padded, -angle, reshape=False, order=1)
        sinogram[i, :] = rotated.sum(axis=0)[:n_det]

    return sinogram


# ─── Inverse Solver: FBP ──────────────────────────────────────────
def fbp_reconstruct(sinogram, angles_deg, img_size):
    """
    Filtered Back-Projection using skimage.transform.iradon with ramp filter.
    sinogram: shape (n_angles, n_det)
    angles_deg: 1-D array of projection angles in degrees
    """
    # iradon expects (n_det, n_angles)
    sino_T = sinogram.T
    recon = iradon(sino_T, theta=angles_deg, filter_name='ramp',
                   output_size=img_size, circle=False)
    return recon


def unfiltered_backproject(sinogram, angles_deg, img_size):
    """
    Unfiltered backprojection using iradon with filter_name=None.
    Used as the adjoint / correction operator in iterative methods.
    """
    sino_T = sinogram.T
    recon = iradon(sino_T, theta=angles_deg, filter_name=None,
                   output_size=img_size, circle=False)
    return recon


# ─── Inverse Solver: SIRT ─────────────────────────────────────────
def sirt_reconstruct(sinogram, angles_deg, img_size, n_iter=50):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT).
    """
    recon = np.zeros((img_size, img_size))

    print(f"  SIRT: {n_iter} iterations ...")

    for it in range(n_iter):
        # Forward project current estimate
        sino_est = radon_transform(recon, angles_deg)

        # Trim to match
        n_det = sinogram.shape[1]
        sino_est_trim = sino_est[:, :n_det]

        # Residual
        residual = sinogram - sino_est_trim

        # Backproject residual (unfiltered — correct SIRT operator)
        correction = unfiltered_backproject(residual, angles_deg, img_size)

        # Update with relaxation
        recon += 0.1 * correction
        recon = np.maximum(recon, 0)

        if (it + 1) % 10 == 0:
            err = np.linalg.norm(residual)
            print(f"    iter {it+1:3d}: residual norm = {err:.4f}")

    return recon


# ─── Inverse Solver: CGLS ─────────────────────────────────────────
def cgls_reconstruct(sinogram, angles_deg, img_size, n_iter=30):
    """
    Conjugate Gradient Least Squares for CT reconstruction.
    Uses forward (Radon) and adjoint (backprojection) operators.
    """
    n_det = sinogram.shape[1]

    # Flatten
    b = sinogram.ravel()

    def A_forward(x):
        img = x.reshape(img_size, img_size)
        sino = radon_transform(img, angles_deg)
        return sino[:, :n_det].ravel()

    def AT_backward(y):
        sino = y.reshape(len(angles_deg), n_det)
        img = unfiltered_backproject(sino, angles_deg, img_size)
        return img.ravel()

    # CGLS
    x = np.zeros(img_size * img_size)
    r = b - A_forward(x)
    s = AT_backward(r)
    p = s.copy()
    gamma = np.dot(s, s)

    print(f"  CGLS: {n_iter} iterations ...")

    for it in range(n_iter):
        Ap = A_forward(p)
        alpha = gamma / max(np.dot(Ap, Ap), 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        s = AT_backward(r)
        gamma_new = np.dot(s, s)
        beta = gamma_new / max(gamma, 1e-30)
        p = s + beta * p
        gamma = gamma_new

        if (it + 1) % 10 == 0:
            print(f"    iter {it+1:3d}: ||r|| = {np.linalg.norm(r):.4f}")

    x = np.maximum(x, 0)
    return x.reshape(img_size, img_size)


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
def visualize_results(gt, sinogram, rec_fbp, rec_sirt, angles, metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(gt, cmap='gray')
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)')

    axes[0, 1].imshow(sinogram, aspect='auto', cmap='gray')
    axes[0, 1].set_title(f'Sinogram ({len(angles)} angles)')
    axes[0, 1].set_xlabel('Detector')
    axes[0, 1].set_ylabel('Angle index')

    axes[0, 2].imshow(rec_fbp / max(rec_fbp.max(), 1e-12), cmap='gray')
    axes[0, 2].set_title('FBP Reconstruction')

    axes[1, 0].imshow(rec_sirt / max(rec_sirt.max(), 1e-12), cmap='gray')
    axes[1, 0].set_title('SIRT Reconstruction')

    err = np.abs(gt / max(gt.max(), 1e-12) - rec_sirt / max(rec_sirt.max(), 1e-12))
    axes[1, 1].imshow(err, cmap='hot')
    axes[1, 1].set_title('|Error| (SIRT)')

    # Profile comparison
    mid = gt.shape[0] // 2
    axes[1, 2].plot(gt[mid, :] / max(gt[mid, :].max(), 1e-12), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_fbp[mid, :] / max(rec_fbp[mid, :].max(), 1e-12),
                     'g--', lw=1.5, label='FBP')
    axes[1, 2].plot(rec_sirt[mid, :] / max(rec_sirt[mid, :].max(), 1e-12),
                     'r--', lw=1.5, label='SIRT')
    axes[1, 2].set_title('Central Profile')
    axes[1, 2].legend()

    fig.suptitle(
        f"TIGRE — Sparse-View CT Reconstruction ({N_ANGLES_SPARSE} views)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  TIGRE — Sparse-View CT Reconstruction")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation — Shepp-Logan Phantom")
    phantom = generate_shepp_logan(IMG_SIZE)
    print(f"  Phantom: {phantom.shape}")
    print(f"  Value range: [{phantom.min():.3f}, {phantom.max():.3f}]")

    # Stage 2: Forward — Radon Transform
    print("\n[STAGE 2] Forward — Radon Transform (Sparse View)")
    angles_sparse = np.linspace(0, 180, N_ANGLES_SPARSE, endpoint=False)
    sinogram_sparse = radon_transform(phantom, angles_sparse)
    # Add Poisson-like noise
    sig_power = np.mean(sinogram_sparse**2)
    noise_power = sig_power / (10**(NOISE_SNR_DB / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(sinogram_sparse.shape)
    sinogram_noisy = sinogram_sparse + noise
    print(f"  Sinogram: {sinogram_noisy.shape} ({N_ANGLES_SPARSE} angles)")

    # Stage 3a: FBP
    print("\n[STAGE 3a] Inverse — Filtered Back-Projection")
    rec_fbp = fbp_reconstruct(sinogram_noisy, angles_sparse, IMG_SIZE)
    rec_fbp = np.maximum(rec_fbp, 0)
    m_fbp = compute_metrics(phantom, rec_fbp)
    print(f"  FBP: CC={m_fbp['CC']:.4f}, PSNR={m_fbp['PSNR']:.1f}")

    # Stage 3b: SIRT
    print("\n[STAGE 3b] Inverse — SIRT")
    rec_sirt = sirt_reconstruct(sinogram_noisy, angles_sparse, IMG_SIZE, n_iter=50)
    m_sirt = compute_metrics(phantom, rec_sirt)
    print(f"  SIRT: CC={m_sirt['CC']:.4f}, PSNR={m_sirt['PSNR']:.1f}")

    # Choose best
    if m_sirt['CC'] >= m_fbp['CC']:
        rec_best = rec_sirt
        metrics = m_sirt
        method = "SIRT"
    else:
        rec_best = rec_fbp
        metrics = m_fbp
        method = "FBP"
    print(f"\n  → Using {method} (higher CC)")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), rec_best)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), phantom)
    # Also save to working dir for website assets
    np.save(os.path.join(WORKING_DIR, "gt_output.npy"), phantom)
    np.save(os.path.join(WORKING_DIR, "recon_output.npy"), rec_best)

    visualize_results(phantom, sinogram_noisy, rec_fbp, rec_sirt,
                      angles_sparse, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
