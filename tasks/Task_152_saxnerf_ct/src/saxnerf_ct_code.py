"""
saxnerf_ct - Sparse-View CT Reconstruction
Task: From limited X-ray projections (sparse angles), reconstruct CT volume
Repo: https://github.com/caiyuanhao1998/SAX-NeRF

This script demonstrates sparse-view CT reconstruction using:
1. Shepp-Logan phantom as ground truth
2. FBP baseline with sparse angles
3. TV-regularized iterative reconstruction (FISTA-TV) for sparse-view recovery
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(WORKING_DIR, "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_phantom(size=128):
    """Generate Shepp-Logan phantom at specified resolution."""
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)
    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-12)
    return phantom


def compute_sinogram(image, angles):
    """Compute sinogram via Radon transform at given angles."""
    return radon(image, theta=angles, circle=True)


def fbp_reconstruction(sinogram, angles, size):
    """Filtered back-projection reconstruction."""
    recon = iradon(sinogram, theta=angles, circle=True,
                   output_size=size, filter_name='ramp')
    return recon


def radon_forward(image, angles):
    """Forward Radon transform operator."""
    return radon(image, theta=angles, circle=True)


def radon_adjoint(sinogram, angles, size):
    """Adjoint (backprojection without filter) operator."""
    return iradon(sinogram, theta=angles, circle=True,
                  output_size=size, filter_name=None)


def tv_proximal(x, weight):
    """TV proximal operator using Chambolle's algorithm."""
    # Clip to avoid issues, then denoise
    x_clipped = np.clip(x, 0, None)
    denoised = denoise_tv_chambolle(x_clipped, weight=weight)
    return denoised


def fista_tv_reconstruction(sinogram_sparse, angles_sparse, size,
                            n_iter=150, tv_weight=0.005, step_size=None):
    """
    FISTA with TV regularization for sparse-view CT reconstruction.

    Solves: min_x  0.5 * ||A*x - y||^2 + lambda * TV(x)
    where A is the Radon transform at sparse angles, y is the sparse sinogram.
    """
    # Estimate step size from operator norm (Lipschitz constant)
    # For Radon, we approximate: L ~ n_angles * size
    if step_size is None:
        # Estimate Lipschitz constant via power iteration
        x_test = np.random.randn(size, size)
        for _ in range(5):
            y_test = radon_forward(x_test, angles_sparse)
            x_test = radon_adjoint(y_test, angles_sparse, size)
            norm_est = np.sqrt(np.sum(x_test ** 2))
            x_test = x_test / norm_est
        step_size = 1.0 / norm_est
        print(f"  Estimated step size: {step_size:.6f} (L={norm_est:.1f})")

    # Initialize with FBP
    x = fbp_reconstruction(sinogram_sparse, angles_sparse, size)
    x = np.clip(x, 0, None)
    x_prev = x.copy()
    t = 1.0

    print(f"  Running FISTA-TV: {n_iter} iterations, tv_weight={tv_weight}")
    for k in range(n_iter):
        # Momentum (FISTA)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        momentum = (t - 1) / t_new
        z = x + momentum * (x - x_prev)

        # Gradient step: gradient of 0.5*||Ax - y||^2 = A^T(Ax - y)
        residual = radon_forward(z, angles_sparse) - sinogram_sparse
        gradient = radon_adjoint(residual, angles_sparse, size)
        z_grad = z - step_size * gradient

        # TV proximal step
        x_new = tv_proximal(z_grad, weight=tv_weight * step_size)

        # Update
        x_prev = x.copy()
        x = x_new
        t = t_new

        if (k + 1) % 30 == 0 or k == 0:
            data_fit = 0.5 * np.sum(residual ** 2)
            print(f"    Iter {k+1:3d}: data_fit = {data_fit:.2f}")

    return np.clip(x, 0, None)


def compute_psnr(gt, recon, data_range=None):
    """Compute PSNR between ground truth and reconstruction."""
    if data_range is None:
        data_range = gt.max() - gt.min()
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_rmse(gt, recon):
    """Compute RMSE between ground truth and reconstruction."""
    return np.sqrt(np.mean((gt - recon) ** 2))


def compute_ssim(gt, recon):
    """Compute SSIM between ground truth and reconstruction."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)


def visualize_results(phantom, fbp_sparse, tv_recon, results_dir):
    """Create 4-panel visualization: GT, Sparse FBP, TV Recon, Error map."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Ground Truth
    im0 = axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Sparse FBP
    fbp_clipped = np.clip(fbp_sparse, 0, 1)
    im1 = axes[0, 1].imshow(fbp_clipped, cmap='gray', vmin=0, vmax=1)
    psnr_fbp = compute_psnr(phantom, fbp_clipped, data_range=1.0)
    ssim_fbp = compute_ssim(phantom, fbp_clipped)
    axes[0, 1].set_title(f'Sparse FBP (30 angles)\nPSNR={psnr_fbp:.1f}dB, SSIM={ssim_fbp:.3f}',
                         fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # TV Reconstruction
    tv_clipped = np.clip(tv_recon, 0, 1)
    im2 = axes[1, 0].imshow(tv_clipped, cmap='gray', vmin=0, vmax=1)
    psnr_tv = compute_psnr(phantom, tv_clipped, data_range=1.0)
    ssim_tv = compute_ssim(phantom, tv_clipped)
    axes[1, 0].set_title(f'FISTA-TV Recon (30 angles)\nPSNR={psnr_tv:.1f}dB, SSIM={ssim_tv:.3f}',
                         fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Error map (TV recon)
    error_map = np.abs(phantom - tv_clipped)
    im3 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title(f'Error Map (|GT - TV Recon|)\nRMSE={compute_rmse(phantom, tv_clipped):.4f}',
                         fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.suptitle('Sparse-View CT Reconstruction\n(SAX-NeRF Task: 30 sparse projections → CT volume)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    print("=" * 60)
    print("Sparse-View CT Reconstruction (SAX-NeRF Task)")
    print("=" * 60)

    # --- Step 1: Generate phantom ---
    print("\n[1/5] Generating Shepp-Logan phantom (128x128)...")
    size = 128
    phantom = generate_phantom(size)
    print(f"  Phantom shape: {phantom.shape}, range: [{phantom.min():.3f}, {phantom.max():.3f}]")

    # --- Step 2: Generate sinograms ---
    print("\n[2/5] Computing sinograms...")
    n_full = 180
    n_sparse = 30
    angles_full = np.linspace(0, 180, n_full, endpoint=False)
    angles_sparse = np.linspace(0, 180, n_sparse, endpoint=False)

    sinogram_full = compute_sinogram(phantom, angles_full)
    sinogram_sparse = compute_sinogram(phantom, angles_sparse)
    print(f"  Full sinogram: {sinogram_full.shape} ({n_full} angles)")
    print(f"  Sparse sinogram: {sinogram_sparse.shape} ({n_sparse} angles)")

    # --- Step 3: FBP reconstructions ---
    print("\n[3/5] FBP reconstruction (baseline)...")
    fbp_full = fbp_reconstruction(sinogram_full, angles_full, size)
    fbp_full = np.clip(fbp_full, 0, 1)
    fbp_sparse = fbp_reconstruction(sinogram_sparse, angles_sparse, size)
    fbp_sparse_clipped = np.clip(fbp_sparse, 0, 1)

    psnr_fbp_full = compute_psnr(phantom, fbp_full, data_range=1.0)
    ssim_fbp_full = compute_ssim(phantom, fbp_full)
    psnr_fbp_sparse = compute_psnr(phantom, fbp_sparse_clipped, data_range=1.0)
    ssim_fbp_sparse = compute_ssim(phantom, fbp_sparse_clipped)

    print(f"  FBP full ({n_full} angles): PSNR={psnr_fbp_full:.2f}dB, SSIM={ssim_fbp_full:.4f}")
    print(f"  FBP sparse ({n_sparse} angles): PSNR={psnr_fbp_sparse:.2f}dB, SSIM={ssim_fbp_sparse:.4f}")

    # --- Step 4: FISTA-TV iterative reconstruction ---
    print(f"\n[4/5] FISTA-TV iterative reconstruction ({n_sparse} sparse angles)...")
    tv_recon = fista_tv_reconstruction(
        sinogram_sparse, angles_sparse, size,
        n_iter=200, tv_weight=0.008, step_size=None
    )
    tv_recon_clipped = np.clip(tv_recon, 0, 1)

    psnr_tv = compute_psnr(phantom, tv_recon_clipped, data_range=1.0)
    ssim_tv = compute_ssim(phantom, tv_recon_clipped)
    rmse_tv = compute_rmse(phantom, tv_recon_clipped)

    print(f"\n  FISTA-TV result: PSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.4f}, RMSE={rmse_tv:.4f}")
    print(f"  Improvement over sparse FBP: "
          f"PSNR +{psnr_tv - psnr_fbp_sparse:.2f}dB, "
          f"SSIM +{ssim_tv - ssim_fbp_sparse:.4f}")

    # --- Step 5: Save results ---
    print("\n[5/5] Saving results...")

    # Save numpy arrays
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), phantom)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), tv_recon_clipped)
    np.save(os.path.join(RESULTS_DIR, 'fbp_sparse.npy'), fbp_sparse_clipped)
    np.save(os.path.join(RESULTS_DIR, 'sinogram_sparse.npy'), sinogram_sparse)
    print("  Saved .npy files")

    # Save metrics
    metrics = {
        "task": "saxnerf_ct",
        "description": "Sparse-view CT reconstruction using FISTA-TV",
        "phantom_size": size,
        "n_full_angles": n_full,
        "n_sparse_angles": n_sparse,
        "fbp_full": {
            "psnr_db": round(psnr_fbp_full, 4),
            "ssim": round(ssim_fbp_full, 4)
        },
        "fbp_sparse": {
            "psnr_db": round(psnr_fbp_sparse, 4),
            "ssim": round(ssim_fbp_sparse, 4)
        },
        "fista_tv": {
            "psnr_db": round(psnr_tv, 4),
            "ssim": round(ssim_tv, 4),
            "rmse": round(rmse_tv, 6),
            "n_iterations": 200,
            "tv_weight": 0.008
        },
        "improvement_over_sparse_fbp": {
            "psnr_gain_db": round(psnr_tv - psnr_fbp_sparse, 4),
            "ssim_gain": round(ssim_tv - ssim_fbp_sparse, 4)
        }
    }

    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Save visualization
    visualize_results(phantom, fbp_sparse_clipped, tv_recon_clipped, RESULTS_DIR)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ground truth: Shepp-Logan phantom {size}x{size}")
    print(f"  Sparse angles: {n_sparse} (of {n_full})")
    print(f"  FBP sparse:  PSNR={psnr_fbp_sparse:.2f}dB, SSIM={ssim_fbp_sparse:.4f}")
    print(f"  FISTA-TV:    PSNR={psnr_tv:.2f}dB, SSIM={ssim_tv:.4f}, RMSE={rmse_tv:.6f}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    metrics = main()
