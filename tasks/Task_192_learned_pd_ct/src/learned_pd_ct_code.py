"""
Task 192: learned_pd_ct — CT Reconstruction via TV-PDHG
=========================================================
Inverse problem: Reconstruct a 2D image from its Radon transform (CT sinogram).

Forward model: y = R(x) + noise
  R: Radon transform (parallel beam CT)
  x: 2D phantom image (ground truth)
  y: noisy sinogram (input measurement)

Inverse solver: Two-stage approach:
  1. Filtered Back-Projection (FBP) for initial reconstruction
  2. Total Variation (TV) denoising via Chambolle-Pock PDHG to remove artifacts

  This mirrors the "unrolled" primal-dual approach: the FBP gives a good initial
  estimate, and TV-PDHG refines it — analogous to a single "block" of the Learned
  Primal-Dual network (Adler & Öktem, IEEE TMI 2018) but with classical TV instead
  of learned CNN components.

Reference: Adler & Öktem, "Learned Primal-Dual Reconstruction", IEEE TMI 2018
Repo: https://github.com/adler-j/learned_primal_dual

Usage: /data/yjh/learned_pd_ct_env/bin/python learned_pd_ct_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon

# ── 1. Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 2. Generate Shepp-Logan phantom ──────────────────────────────────────────
N = 256
gt = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True).astype('float64')
gt = (gt - gt.min()) / (gt.max() - gt.min())  # normalise to [0, 1]
print(f"[INFO] Phantom shape: {gt.shape}, range: [{gt.min():.4f}, {gt.max():.4f}]")

# ── 3. Forward model: Radon transform + noise ────────────────────────────────
n_angles = 180
theta_angles = np.linspace(0., 180., n_angles, endpoint=False)

# Clean sinogram
sinogram_clean = radon(gt, theta=theta_angles, circle=False)

# Add Gaussian noise (1% of peak sinogram value)
noise_std = 0.01 * sinogram_clean.max()
rng = np.random.default_rng(42)
noise = noise_std * rng.standard_normal(sinogram_clean.shape)
sinogram_noisy = sinogram_clean + noise

print(f"[INFO] Sinogram shape: {sinogram_noisy.shape}")
print(f"[INFO] Noise std: {noise_std:.4f} (1% of sinogram max {sinogram_clean.max():.2f})")

# ── 4. Stage 1: Filtered Back-Projection (FBP) ──────────────────────────────
fbp_recon = iradon(sinogram_noisy, theta=theta_angles, circle=False,
                   filter_name='ramp')
fbp_recon = np.clip(fbp_recon, 0, 1)

fbp_psnr = psnr_fn(gt, fbp_recon, data_range=1.0)
print(f"[INFO] FBP reconstruction: PSNR = {fbp_psnr:.2f} dB")

# ── 5. Stage 2: TV-PDHG denoising (Chambolle-Pock) ──────────────────────────
#
# Solve:  min_x  0.5 * ||x - x_fbp||^2  +  lam * ||∇x||_{2,1}
#         s.t.   0 <= x <= 1
#
# This is a TV-denoising (ROF) problem solved by PDHG:
#   min_x  f(x) + g(∇x)
#   f(x) = 0.5 * ||x - x_fbp||^2 + indicator_{[0,1]}(x)
#   g(y) = lam * ||y||_{2,1}
#
# Proximal operators:
#   prox_{tau*f}(z) = clip((z + tau*x_fbp) / (1+tau), 0, 1)
#   prox_{sigma*g*}(p) = pointwise project onto l2 ball of radius lam


def grad(x):
    """Discrete gradient with forward differences: returns (2, N, N)."""
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:-1, :] = x[1:, :] - x[:-1, :]
    gy[:, :-1] = x[:, 1:] - x[:, :-1]
    return np.stack([gx, gy], axis=0)


def div(p):
    """Discrete divergence = -∇^* (negative adjoint of gradient)."""
    px, py = p[0], p[1]
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)
    dx[0, :] = px[0, :]
    dx[1:-1, :] = px[1:-1, :] - px[:-2, :]
    dx[-1, :] = -px[-2, :]
    dy[:, 0] = py[:, 0]
    dy[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
    dy[:, -1] = -py[:, -1]
    return dx + dy


# ||∇||^2 <= 8 for 2D forward differences
norm_grad = np.sqrt(8.0)

# TV regularization weight
lam = 0.02

# Step sizes (Chambolle-Pock for denoising)
tau = 1.0 / norm_grad
sigma = 1.0 / norm_grad

niter_tv = 300
print(f"[INFO] TV-PDHG denoising: lam={lam}, niter={niter_tv}")

# Initialize from FBP
x = fbp_recon.copy()
x_bar = x.copy()
p = np.zeros((2, N, N), dtype='float64')  # dual variable

for k in range(niter_tv):
    x_old = x.copy()

    # Dual update: p = prox_{sigma * g*}(p + sigma * ∇(x_bar))
    p = p + sigma * grad(x_bar)
    # Project onto l2 balls of radius lam
    norms = np.sqrt(p[0]**2 + p[1]**2)
    scale = np.maximum(norms / lam, 1.0)
    p = p / scale[np.newaxis, :, :]

    # Primal update: x = prox_{tau * f}(x - tau * (-div(p)))
    #   prox_{tau*f}(z) = clip( (z + tau * x_fbp) / (1 + tau), 0, 1 )
    #   ∇^*(p) = -div(p), so the update is x - tau * (-div(p)) = x + tau * div(p)
    x = np.clip((x + tau * div(p) + tau * fbp_recon) / (1.0 + tau), 0, 1)

    # Over-relaxation (theta=1)
    x_bar = 2 * x - x_old

    if (k + 1) % 50 == 0:
        psnr_k = psnr_fn(gt, x, data_range=1.0)
        print(f"  TV iter {k+1:4d}/{niter_tv}: PSNR = {psnr_k:.2f} dB")

recon = x.astype('float32')
gt32 = gt.astype('float32')
print(f"[INFO] Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")

# ── 6. Evaluation ────────────────────────────────────────────────────────────
data_range = float(gt32.max() - gt32.min())
psnr_val = float(psnr_fn(gt32, recon, data_range=data_range))
ssim_val = float(ssim_fn(gt32, recon, data_range=data_range))
cc_val = float(np.corrcoef(gt32.ravel(), recon.ravel())[0, 1])

print(f"\n{'='*50}")
print(f"  PSNR  = {psnr_val:.2f} dB")
print(f"  SSIM  = {ssim_val:.4f}")
print(f"  CC    = {cc_val:.4f}")
print(f"{'='*50}\n")

# ── 7. Save numerical results ────────────────────────────────────────────────
metrics = {
    "psnr_db": round(psnr_val, 2),
    "ssim": round(ssim_val, 4),
    "cc": round(cc_val, 4),
}
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as fp:
    json.dump(metrics, fp, indent=2)

np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt32)
np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon)
np.save(os.path.join(RESULTS_DIR, "input.npy"), sinogram_noisy.astype('float32'))
print("[INFO] Saved metrics.json, ground_truth.npy, reconstruction.npy, input.npy")

# ── 8. Visualization (2×2 panel) ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Ground truth
im0 = axes[0, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title("(a) Ground Truth Phantom", fontsize=13)
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

# (b) Noisy sinogram
im1 = axes[0, 1].imshow(sinogram_noisy, cmap='gray', aspect='auto')
axes[0, 1].set_title("(b) Noisy Sinogram (Input)", fontsize=13)
axes[0, 1].set_xlabel("Detector pixel")
axes[0, 1].set_ylabel("Projection angle index")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# (c) TV-PDHG reconstruction
im2 = axes[1, 0].imshow(recon, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title(
    f"(c) FBP + TV-PDHG Reconstruction\nPSNR={psnr_val:.1f} dB, SSIM={ssim_val:.3f}",
    fontsize=13,
)
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# (d) Error map
error_map = np.abs(gt32 - recon)
im3 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0,
                         vmax=max(error_map.max(), 0.01))
axes[1, 1].set_title("(d) Absolute Error |GT − Recon|", fontsize=13)
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.suptitle(
    "Task 192: CT Reconstruction via FBP + TV-PDHG\n"
    f"256×256 Shepp-Logan, {n_angles} angles, 1% noise",
    fontsize=15,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[INFO] Saved figure → {fig_path}")

print("\n[DONE] Task 192 completed successfully.")
