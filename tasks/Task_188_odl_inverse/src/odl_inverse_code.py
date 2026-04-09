#!/usr/bin/env python
"""
Task 188: odl_inverse — CT Reconstruction using ODL
=====================================================
Parallel-beam CT reconstruction of the Shepp-Logan phantom using the ODL
(Operator Discretization Library) framework.

Methods:
  1. FBP  — Filtered Back Projection (analytic baseline)
  2. CGLS — Conjugate Gradient Least Squares (iterative)
  3. TV-PDHG — Total Variation regularisation via Primal-Dual Hybrid Gradient

Metrics (PSNR / SSIM) are computed against the ground-truth phantom.
The TV-PDHG result is used as the primary reconstruction output.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import odl

warnings.filterwarnings('ignore')

# ─── output directory ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Reconstruction space & phantom ──────────────────────────────────────
N = 256  # image size
reco_space = odl.uniform_discr(
    min_pt=[-1, -1], max_pt=[1, 1], shape=[N, N], dtype='float32'
)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
gt = phantom.asarray()

# ─── 2. Forward operator (parallel-beam geometry) ───────────────────────────
num_angles = 180
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=num_angles)
ray_transform = odl.tomo.RayTransform(reco_space, geometry)

# ─── 3. Generate noisy sinogram ─────────────────────────────────────────────
sinogram_clean = ray_transform(phantom)
noise_level = 0.02  # relative noise
noise = odl.phantom.white_noise(ray_transform.range)
sinogram = sinogram_clean + noise * np.mean(np.abs(sinogram_clean.asarray())) * noise_level

# ─── 4a. FBP reconstruction ─────────────────────────────────────────────────
print('[FBP] Running Filtered Back Projection …')
fbp_op = odl.tomo.fbp_op(ray_transform)
x_fbp = fbp_op(sinogram)
recon_fbp = x_fbp.asarray()

# ─── 4b. CGLS reconstruction ────────────────────────────────────────────────
print('[CGLS] Running Conjugate Gradient (normal equations) …')
x_cgls = ray_transform.domain.zero()
odl.solvers.conjugate_gradient_normal(
    ray_transform, x_cgls, sinogram, niter=30,
    callback=odl.solvers.CallbackPrintIteration(step=10, fmt='  CGLS iter {}')
)
recon_cgls = x_cgls.asarray()

# ─── 4c. TV-PDHG reconstruction ─────────────────────────────────────────────
print('[TV-PDHG] Running TV-regularised PDHG (warm-started from FBP) …')
gradient = odl.Gradient(reco_space)

# Broadcast operator: K = [ray_transform; gradient]
op = odl.BroadcastOperator(ray_transform, gradient)

# Primal functional: f = 0 (no explicit constraint on x)
f = odl.solvers.ZeroFunctional(op.domain)

# Dual functionals
# Data fidelity: 0.5 * ||Ax - b||_2^2
l2_term = odl.solvers.L2NormSquared(ray_transform.range).translated(sinogram)
# TV penalty: lambda * ||grad(x)||_1  (isotropic)
tv_lambda = 0.0002
l1_term = tv_lambda * odl.solvers.L1Norm(gradient.range)

g = odl.solvers.SeparableSum(l2_term, l1_term)

# Step sizes (tau * sigma * ||K||^2 < 1 required for convergence)
op_norm = 1.1 * odl.power_method_opnorm(op, maxiter=20)
tau = 1.0 / op_norm
sigma = 1.0 / op_norm

# Warm-start from FBP for faster convergence
x_pdhg = x_fbp.copy()
niter_pdhg = 600

odl.solvers.pdhg(
    x_pdhg, f, g, op,
    niter=niter_pdhg, tau=tau, sigma=sigma,
    callback=odl.solvers.CallbackPrintIteration(step=100, fmt='  PDHG iter {}')
)
recon_pdhg = x_pdhg.asarray()

# ─── 5. Quantitative evaluation ─────────────────────────────────────────────
def evaluate(gt_arr, recon_arr, label):
    """Compute PSNR and SSIM between ground truth and reconstruction."""
    data_range = gt_arr.max() - gt_arr.min()
    psnr = float(peak_signal_noise_ratio(gt_arr, recon_arr, data_range=data_range))
    ssim = float(structural_similarity(gt_arr, recon_arr, data_range=data_range))
    print(f'  {label:12s}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}')
    return psnr, ssim

print('\n=== Evaluation ===')
psnr_fbp,  ssim_fbp  = evaluate(gt, recon_fbp,  'FBP')
psnr_cgls, ssim_cgls = evaluate(gt, recon_cgls, 'CGLS')
psnr_pdhg, ssim_pdhg = evaluate(gt, recon_pdhg, 'TV-PDHG')

# ─── 6. Save outputs ────────────────────────────────────────────────────────
# Primary reconstruction = TV-PDHG
np.save(os.path.join(OUT_DIR, 'ground_truth.npy'), gt)
np.save(os.path.join(OUT_DIR, 'reconstruction.npy'), recon_pdhg)

metrics = {
    'task': 'odl_inverse',
    'method': 'TV-PDHG (Total Variation via Primal-Dual Hybrid Gradient)',
    'PSNR': round(psnr_pdhg, 4),
    'SSIM': round(ssim_pdhg, 4),
    'all_methods': {
        'FBP':     {'PSNR': round(psnr_fbp,  4), 'SSIM': round(ssim_fbp,  4)},
        'CGLS':    {'PSNR': round(psnr_cgls, 4), 'SSIM': round(ssim_cgls, 4)},
        'TV-PDHG': {'PSNR': round(psnr_pdhg, 4), 'SSIM': round(ssim_pdhg, 4)},
    },
    'parameters': {
        'image_size': N,
        'num_angles': num_angles,
        'noise_level': noise_level,
        'tv_lambda': tv_lambda,
        'pdhg_iterations': niter_pdhg,
    }
}
with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as fp:
    json.dump(metrics, fp, indent=2)
print(f'\nMetrics saved → {os.path.join(OUT_DIR, "metrics.json")}')

# ─── 7. Visualisation ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: images
ax = axes[0, 0]
im = ax.imshow(gt, cmap='gray', vmin=gt.min(), vmax=gt.max())
ax.set_title('Ground Truth (Shepp-Logan)', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[0, 1]
im = ax.imshow(sinogram.asarray(), cmap='gray', aspect='auto')
ax.set_title(f'Sinogram ({num_angles} angles, {noise_level*100:.0f}% noise)', fontsize=12)
ax.set_xlabel('Detector pixel')
ax.set_ylabel('Angle index')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[0, 2]
im = ax.imshow(recon_fbp, cmap='gray', vmin=gt.min(), vmax=gt.max())
ax.set_title(f'FBP  (PSNR={psnr_fbp:.1f} dB, SSIM={ssim_fbp:.3f})', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1, 0]
im = ax.imshow(recon_cgls, cmap='gray', vmin=gt.min(), vmax=gt.max())
ax.set_title(f'CGLS  (PSNR={psnr_cgls:.1f} dB, SSIM={ssim_cgls:.3f})', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1, 1]
im = ax.imshow(recon_pdhg, cmap='gray', vmin=gt.min(), vmax=gt.max())
ax.set_title(f'TV-PDHG  (PSNR={psnr_pdhg:.1f} dB, SSIM={ssim_pdhg:.3f})', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Error map for TV-PDHG
ax = axes[1, 2]
err = np.abs(gt - recon_pdhg)
im = ax.imshow(err, cmap='hot')
ax.set_title('|GT − TV-PDHG| Error Map', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Task 188: ODL Inverse — CT Reconstruction Comparison', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Figure saved  → {os.path.join(OUT_DIR, "reconstruction_result.png")}')

print('\n✓ Task 188 (odl_inverse) completed successfully.')
print(f'  Primary result (TV-PDHG): PSNR={psnr_pdhg:.2f} dB, SSIM={ssim_pdhg:.4f}')
