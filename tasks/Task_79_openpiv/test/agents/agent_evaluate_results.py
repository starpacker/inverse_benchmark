import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(gt_u, gt_v, recon_u, recon_v, frame_a, frame_b,
                     x_grid, y_grid, results_dir):
    """
    Compute PIV-specific evaluation metrics and generate visualizations.
    
    Args:
        gt_u: Ground truth u velocity field
        gt_v: Ground truth v velocity field
        recon_u: Reconstructed u velocity field
        recon_v: Reconstructed v velocity field
        frame_a: First frame
        frame_b: Second frame
        x_grid: x coordinates of PIV grid
        y_grid: y coordinates of PIV grid
        results_dir: Directory to save results
    
    Returns:
        dict of metrics including:
        - rmse_u, rmse_v: RMSE for each component
        - rmse_magnitude: RMSE for velocity magnitude
        - cc_u, cc_v: Correlation coefficients
        - re: Relative error
        - psnr: PSNR based on velocity magnitude range
        - aee: Average Endpoint Error
        - ssim: Structural similarity
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Velocity magnitudes
    gt_mag = np.sqrt(gt_u**2 + gt_v**2)
    recon_mag = np.sqrt(recon_u**2 + recon_v**2)
    error_mag = np.abs(gt_mag - recon_mag)
    
    # RMSE per component
    rmse_u = np.sqrt(np.mean((gt_u - recon_u)**2))
    rmse_v = np.sqrt(np.mean((gt_v - recon_v)**2))
    
    # RMSE of magnitude
    rmse_mag = np.sqrt(np.mean((gt_mag - recon_mag)**2))
    
    # Correlation coefficient
    cc_u = np.corrcoef(gt_u.flatten(), recon_u.flatten())[0, 1]
    cc_v = np.corrcoef(gt_v.flatten(), recon_v.flatten())[0, 1]
    cc_mag = np.corrcoef(gt_mag.flatten(), recon_mag.flatten())[0, 1]
    
    # Relative Error
    gt_norm = np.sqrt(np.mean(gt_u**2 + gt_v**2))
    error_norm = np.sqrt(np.mean((gt_u - recon_u)**2 + (gt_v - recon_v)**2))
    re = error_norm / gt_norm if gt_norm > 0 else float('inf')
    
    # Average Endpoint Error
    aee = np.mean(np.sqrt((gt_u - recon_u)**2 + (gt_v - recon_v)**2))
    
    # PSNR based on velocity magnitude
    data_range = gt_mag.max() - gt_mag.min()
    mse = np.mean((gt_mag - recon_mag)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # SSIM of velocity magnitude
    gt_mag_norm = (gt_mag - gt_mag.min()) / (gt_mag.max() - gt_mag.min() + 1e-10)
    recon_mag_norm = (recon_mag - recon_mag.min()) / (recon_mag.max() - recon_mag.min() + 1e-10)
    min_dim = min(gt_mag_norm.shape)
    win_size = min(7, min_dim) if min_dim >= 3 else 3
    if win_size % 2 == 0:
        win_size -= 1
    ssim_val = ssim(gt_mag_norm, recon_mag_norm, data_range=1.0, win_size=win_size)
    
    metrics = {
        'rmse_u': float(rmse_u),
        'rmse_v': float(rmse_v),
        'rmse_magnitude': float(rmse_mag),
        'cc_u': float(cc_u),
        'cc_v': float(cc_v),
        'cc_magnitude': float(cc_mag),
        'relative_error': float(re),
        'average_endpoint_error': float(aee),
        'psnr': float(psnr),
        'ssim': float(ssim_val),
        'rmse': float(rmse_mag),
    }
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    gt_velocity = np.stack([gt_u, gt_v], axis=0)
    recon_velocity = np.stack([recon_u, recon_v], axis=0)
    input_data = np.stack([frame_a, frame_b], axis=0)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_velocity)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_velocity)
    np.save(os.path.join(results_dir, "input.npy"), input_data)
    print(f"[SAVE] Ground truth shape: {gt_velocity.shape} → ground_truth.npy")
    print(f"[SAVE] Reconstruction shape: {recon_velocity.shape} → reconstruction.npy")
    print(f"[SAVE] Input shape: {input_data.shape} → input.npy")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Images and velocity fields
    axes[0, 0].imshow(frame_a, cmap='gray')
    axes[0, 0].set_title('Frame A (Particle Image)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(frame_b, cmap='gray')
    axes[0, 1].set_title('Frame B (Displaced Particles)', fontsize=12)
    axes[0, 1].axis('off')
    
    im_gt = axes[0, 2].imshow(gt_mag, cmap='jet', origin='upper')
    axes[0, 2].set_title('Ground Truth |V|', fontsize=12)
    plt.colorbar(im_gt, ax=axes[0, 2], fraction=0.046, label='pixels/frame')
    
    im_recon = axes[0, 3].imshow(recon_mag, cmap='jet', origin='upper',
                                  vmin=gt_mag.min(), vmax=gt_mag.max())
    axes[0, 3].set_title('Reconstructed |V|', fontsize=12)
    plt.colorbar(im_recon, ax=axes[0, 3], fraction=0.046, label='pixels/frame')
    
    # Row 2: Quiver plots and error
    skip = 1
    axes[1, 0].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                       gt_u[::skip, ::skip], -gt_v[::skip, ::skip],
                       gt_mag[::skip, ::skip], cmap='jet',
                       scale=None, width=0.004)
    axes[1, 0].set_title('GT Velocity Field', fontsize=12)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                       recon_u[::skip, ::skip], -recon_v[::skip, ::skip],
                       recon_mag[::skip, ::skip], cmap='jet',
                       scale=None, width=0.004)
    axes[1, 1].set_title('Reconstructed Velocity Field', fontsize=12)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    
    im_err = axes[1, 2].imshow(error_mag, cmap='hot', origin='upper')
    axes[1, 2].set_title('Velocity Magnitude Error', fontsize=12)
    plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, label='pixels/frame')
    
    axes[1, 3].scatter(gt_mag.flatten(), recon_mag.flatten(), alpha=0.5, s=10, c='steelblue')
    max_val = max(gt_mag.max(), recon_mag.max()) * 1.1
    axes[1, 3].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Identity')
    axes[1, 3].set_xlabel('GT |V| (px/frame)', fontsize=11)
    axes[1, 3].set_ylabel('Recon |V| (px/frame)', fontsize=11)
    axes[1, 3].set_title(f'GT vs Recon (CC={metrics["cc_magnitude"]:.4f})', fontsize=12)
    axes[1, 3].legend()
    axes[1, 3].set_aspect('equal')
    axes[1, 3].set_xlim([0, max_val])
    axes[1, 3].set_ylim([0, max_val])
    
    fig.suptitle(
        f"OpenPIV — PIV Flow Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"AEE={metrics['average_endpoint_error']:.4f} px | "
        f"CC={metrics['cc_magnitude']:.4f} | RE={metrics['relative_error']:.4f}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics
