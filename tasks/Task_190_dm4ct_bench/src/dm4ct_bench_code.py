"""
dm4ct_bench - Diffusion Model CT Reconstruction Benchmark
=========================================================
Task: Sparse-view CT reconstruction with diffusion-style iterative refinement
Repo: https://github.com/DM4CT/DM4CT

Implements a sparse-view CT reconstruction pipeline inspired by the DM4CT
benchmark framework. Uses iterative data-consistency refinement with
learned denoising to enhance FBP reconstructions from undersampled sinograms.

Usage:
    /data/yjh/dm4ct_bench_env/bin/python dm4ct_bench_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import time

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 256
N_ANGLES_FULL = 180
N_ANGLES_SPARSE = 30  # Sparse-view: 30 angles
NOISE_LEVEL = 0.02
TV_WEIGHT = 0.005
N_OUTER_ITER = 60  # Iterative refinement iterations
N_TV_ITER = 80     # TV denoising sub-iterations

# ═══════════════════════════════════════════════════════════
# 2. Data Generation - Shepp-Logan Phantom + CT
# ═══════════════════════════════════════════════════════════
def shepp_logan_phantom(size=256):
    """Generate modified Shepp-Logan phantom."""
    from skimage.data import shepp_logan_phantom as slp
    from skimage.transform import resize
    phantom = slp()
    if phantom.shape[0] != size:
        phantom = resize(phantom, (size, size), anti_aliasing=True)
    return phantom.astype(np.float64)


def radon_transform(image, angles):
    """Compute Radon transform (sinogram)."""
    from skimage.transform import radon
    return radon(image, theta=angles, circle=True)


def fbp_reconstruct(sinogram, angles, size):
    """Filtered Back Projection reconstruction."""
    from skimage.transform import iradon
    recon = iradon(sinogram, theta=angles, circle=True, filter_name='ramp')
    # Resize if needed
    if recon.shape[0] != size:
        from skimage.transform import resize
        recon = resize(recon, (size, size), anti_aliasing=True)
    return recon


# ═══════════════════════════════════════════════════════════
# 3. TV Denoising (Proximal Operator)
# ═══════════════════════════════════════════════════════════
def tv_denoise(image, weight, n_iter=50):
    """
    Total Variation denoising using Chambolle's projection algorithm.
    Acts as the 'denoiser' in the diffusion-style iterative framework.
    """
    from skimage.restoration import denoise_tv_chambolle
    return denoise_tv_chambolle(image, weight=weight, max_num_iter=n_iter)


# ═══════════════════════════════════════════════════════════
# 4. Diffusion-Style Iterative CT Reconstruction
# ═══════════════════════════════════════════════════════════
def data_consistency_step(image, sinogram, angles, step_size=0.1):
    """
    Data consistency: project current estimate, compute residual,
    back-project residual to enforce measurement consistency.
    Uses unfiltered back-projection for gradient computation.
    """
    from skimage.transform import radon, iradon
    
    # Forward project current estimate
    sino_est = radon(image, theta=angles, circle=True)
    
    # Residual in sinogram domain
    residual_sino = sinogram - sino_est
    
    # Back-project residual WITHOUT filter (gradient of data fidelity)
    correction = iradon(residual_sino, theta=angles, circle=True, filter_name=None)
    
    # Resize if needed
    if correction.shape != image.shape:
        from skimage.transform import resize
        correction = resize(correction, image.shape, anti_aliasing=True)
    
    # Normalize correction
    if np.max(np.abs(correction)) > 0:
        correction = correction / np.max(np.abs(correction)) * step_size
    
    # Apply correction
    return image + correction


def diffusion_style_ct_reconstruction(sinogram, angles, image_size, 
                                       n_outer=15, tv_weight=0.002, 
                                       dc_step_size=0.3):
    """
    Diffusion-style iterative CT reconstruction:
    1. Initialize with FBP
    2. For each iteration:
       a. TV denoising (acts as learned prior/denoiser proxy)
       b. Data consistency (gradient step to match measurements)
    3. Return refined reconstruction
    
    This mirrors the structure of diffusion-based CT methods:
    - Denoising step ≈ score function / learned prior
    - Data consistency ≈ likelihood/measurement constraint
    """
    # Initialize with FBP
    x = fbp_reconstruct(sinogram, angles, image_size)
    x_fbp = x.copy()
    
    print(f"[RECON] Starting diffusion-style iterative refinement...")
    print(f"  Config: {n_outer} outer iters, TV weight={tv_weight}, DC step={dc_step_size}")
    
    # Adaptive TV weight schedule (decrease over iterations, like noise schedule)
    tv_schedule = np.linspace(tv_weight * 2, tv_weight * 0.5, n_outer)
    dc_schedule = np.linspace(dc_step_size * 0.3, dc_step_size * 0.8, n_outer)
    
    for i in range(n_outer):
        # Step 1: Denoise (prior/score function proxy)
        x_denoised = tv_denoise(x, weight=tv_schedule[i], n_iter=N_TV_ITER)
        
        # Step 2: Data consistency
        x = data_consistency_step(x_denoised, sinogram, angles, step_size=dc_schedule[i])
        
        # Clip to valid range
        x = np.clip(x, 0, 1)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Iter {i+1}/{n_outer}: TV_w={tv_schedule[i]:.5f}, range=[{x.min():.3f}, {x.max():.3f}]")
    
    return x, x_fbp


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_psnr(ref, test, data_range=None):
    """Compute PSNR (dB)."""
    if data_range is None:
        data_range = ref.max() - ref.min()
    mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)

def compute_ssim(ref, test):
    """Compute SSIM."""
    from skimage.metrics import structural_similarity as ssim
    data_range = ref.max() - ref.min()
    if data_range == 0:
        data_range = 1.0
    return ssim(ref, test, data_range=data_range)

def compute_rmse(ref, test):
    """Compute RMSE."""
    return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(ground_truth, sinogram, fbp_recon, diffusion_recon, 
                     metrics_fbp, metrics_diff, save_path):
    """Generate 5-panel visualization."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Panel 1: Ground Truth
    axes[0].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    # Panel 2: Sparse Sinogram
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f'Sparse Sinogram\n({N_ANGLES_SPARSE} views)', fontsize=12)
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('Detector')
    
    # Panel 3: FBP Baseline
    axes[2].imshow(np.clip(fbp_recon, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'FBP Baseline\nPSNR={metrics_fbp["psnr"]:.1f}dB', fontsize=12)
    axes[2].axis('off')
    
    # Panel 4: Diffusion-style Reconstruction
    axes[3].imshow(np.clip(diffusion_recon, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'Iterative Recon\nPSNR={metrics_diff["psnr"]:.1f}dB', fontsize=12)
    axes[3].axis('off')
    
    # Panel 5: Error Map
    error = np.abs(ground_truth - diffusion_recon)
    axes[4].imshow(error, cmap='hot', vmin=0, vmax=0.3)
    axes[4].set_title('|Error|', fontsize=12)
    axes[4].axis('off')
    
    fig.suptitle(
        f"DM4CT Sparse-View CT | Diffusion-Style: PSNR={metrics_diff['psnr']:.2f}dB, "
        f"SSIM={metrics_diff['ssim']:.4f} | FBP: PSNR={metrics_fbp['psnr']:.2f}dB",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  dm4ct_bench — Diffusion-Style CT Reconstruction")
    print("=" * 60)
    
    t0 = time.time()
    
    # (a) Generate phantom and sinogram
    print("\n[DATA] Generating Shepp-Logan phantom...")
    phantom = shepp_logan_phantom(IMAGE_SIZE)
    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-10)
    print(f"[DATA] Phantom shape: {phantom.shape}, range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    
    # Full-view sinogram (reference)
    angles_full = np.linspace(0, 180, N_ANGLES_FULL, endpoint=False)
    sino_full = radon_transform(phantom, angles_full)
    
    # Sparse-view sinogram
    angles_sparse = np.linspace(0, 180, N_ANGLES_SPARSE, endpoint=False)
    sino_sparse = radon_transform(phantom, angles_sparse)
    
    # Add noise
    noise = np.random.randn(*sino_sparse.shape) * NOISE_LEVEL * sino_sparse.max()
    sino_noisy = sino_sparse + noise
    
    print(f"[DATA] Sinogram shape: {sino_noisy.shape} ({N_ANGLES_SPARSE} angles)")
    
    # (b) Reconstruct
    recon_diffusion, recon_fbp = diffusion_style_ct_reconstruction(
        sino_noisy, angles_sparse, IMAGE_SIZE,
        n_outer=N_OUTER_ITER, tv_weight=TV_WEIGHT, dc_step_size=0.15
    )
    
    # (c) Evaluate
    metrics_fbp = {
        "psnr": float(compute_psnr(phantom, np.clip(recon_fbp, 0, 1))),
        "ssim": float(compute_ssim(phantom, np.clip(recon_fbp, 0, 1))),
        "rmse": float(compute_rmse(phantom, np.clip(recon_fbp, 0, 1))),
    }
    metrics_diff = {
        "psnr": float(compute_psnr(phantom, np.clip(recon_diffusion, 0, 1))),
        "ssim": float(compute_ssim(phantom, np.clip(recon_diffusion, 0, 1))),
        "rmse": float(compute_rmse(phantom, np.clip(recon_diffusion, 0, 1))),
    }
    
    print(f"\n[EVAL] FBP Baseline:  PSNR={metrics_fbp['psnr']:.2f}dB, SSIM={metrics_fbp['ssim']:.4f}")
    print(f"[EVAL] Diffusion-CT:  PSNR={metrics_diff['psnr']:.2f}dB, SSIM={metrics_diff['ssim']:.4f}")
    print(f"[EVAL] Improvement:   ΔPSNR={metrics_diff['psnr']-metrics_fbp['psnr']:+.2f}dB, "
          f"ΔSSIM={metrics_diff['ssim']-metrics_fbp['ssim']:+.4f}")
    
    # (d) Save metrics
    metrics = {
        "psnr": metrics_diff["psnr"],
        "ssim": metrics_diff["ssim"],
        "rmse": metrics_diff["rmse"],
        "fbp_psnr": metrics_fbp["psnr"],
        "fbp_ssim": metrics_fbp["ssim"],
        "n_angles_sparse": N_ANGLES_SPARSE,
        "n_angles_full": N_ANGLES_FULL,
        "noise_level": NOISE_LEVEL,
        "n_iterations": N_OUTER_ITER,
        "method": "Diffusion-style iterative refinement (TV prior + data consistency)",
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")
    
    # (e) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(phantom, sino_noisy, recon_fbp, recon_diffusion,
                     metrics_fbp, metrics_diff, vis_path)
    
    # (f) Save arrays
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_diffusion)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), phantom)
    
    elapsed = time.time() - t0
    print(f"\n[TIME] Total elapsed: {elapsed:.1f}s")
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
