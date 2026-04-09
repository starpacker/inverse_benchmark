import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from scipy.ndimage import gaussian_filter, maximum_filter, label

def evaluate_results(data_dict, result_dict):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR, SSIM, defect position error, and generates visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary from load_and_preprocess_data.
    result_dict : dict
        Result dictionary from run_inversion.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    from skimage.metrics import structural_similarity
    
    gt_map = data_dict['gt_map']
    recon_norm = result_dict['recon_image']
    defects_m = data_dict['defects_m']
    defects_mm = data_dict['defects_mm']
    x_grid = data_dict['x_grid']
    z_grid = data_dict['z_grid']
    results_dir = data_dict['results_dir']
    recon_time = result_dict['recon_time']
    
    print("\n[3/4] Computing metrics...")
    
    # PSNR
    mse = np.mean((gt_map - recon_norm) ** 2)
    if mse < 1e-20:
        psnr = 100.0
    else:
        data_range = gt_map.max() - gt_map.min()
        psnr = 10 * np.log10(data_range ** 2 / mse)
    
    # SSIM
    ssim = structural_similarity(gt_map, recon_norm, data_range=gt_map.max() - gt_map.min())
    
    # Defect position error
    n_defects = len(defects_m)
    
    # Find local maxima
    filtered = maximum_filter(recon_norm, size=7)
    local_max = (recon_norm == filtered) & (recon_norm > 0.3 * recon_norm.max())
    labeled, n_found = label(local_max)
    
    # Centroid of each region
    peaks = []
    for i in range(1, n_found + 1):
        mask = labeled == i
        coords = np.argwhere(mask)
        iz_mean = coords[:, 0].mean()
        ix_mean = coords[:, 1].mean()
        amplitude = recon_norm[mask].max()
        peaks.append((iz_mean, ix_mean, amplitude))
    
    # Sort by amplitude descending, take top-N
    peaks.sort(key=lambda p: -p[2])
    peaks = peaks[:n_defects]
    
    # Convert to physical coordinates
    peak_positions = []
    for iz, ix, _ in peaks:
        iz_int = int(round(iz))
        ix_int = int(round(ix))
        iz_int = np.clip(iz_int, 0, len(z_grid) - 1)
        ix_int = np.clip(ix_int, 0, len(x_grid) - 1)
        peak_positions.append((x_grid[ix_int], z_grid[iz_int]))
    
    # Greedy nearest-neighbor matching
    gt_remaining = list(defects_m)
    total_error = 0.0
    matched = 0
    for px, pz in peak_positions:
        if not gt_remaining:
            break
        dists = [np.sqrt((px - gx) ** 2 + (pz - gz) ** 2) for gx, gz in gt_remaining]
        best_idx = np.argmin(dists)
        total_error += dists[best_idx]
        gt_remaining.pop(best_idx)
        matched += 1
    
    if matched == 0:
        pos_error_mm = float('inf')
    else:
        pos_error_mm = (total_error / matched) * 1000
    
    print(f"  PSNR:  {psnr:.2f} dB")
    print(f"  SSIM:  {ssim:.4f}")
    print(f"  Mean position error: {pos_error_mm:.2f} mm "
          f"({matched}/{len(defects_m)} defects matched)")
    
    metrics = {
        "task": "arim_ndt",
        "method": "Total Focusing Method (TFM)",
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 4),
        "mean_position_error_mm": round(pos_error_mm, 2),
        "defects_matched": matched,
        "defects_total": len(defects_m),
        "n_elements": data_dict['n_elements'],
        "pitch_mm": data_dict['pitch'] * 1e3,
        "frequency_mhz": data_dict['freq'] / 1e6,
        "sound_speed_m_s": data_dict['c_sound'],
        "grid_nx": data_dict['nx'],
        "grid_nz": data_dict['nz'],
        "snr_db": data_dict['snr_db'],
        "reconstruction_time_s": round(recon_time, 1),
    }
    
    # Save outputs
    print("\n[4/4] Saving results...")
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_map)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_norm)
    print(f"  Saved ground_truth.npy  shape={gt_map.shape}")
    print(f"  Saved reconstruction.npy  shape={recon_norm.shape}")
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")
    
    # Visualization
    x_mm = x_grid * 1e3
    z_mm = z_grid * 1e3
    extent = [x_mm.min(), x_mm.max(), z_mm.max(), z_mm.min()]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Ground Truth
    ax = axes[0]
    im0 = ax.imshow(gt_map, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    for dx_mm, dz_mm in defects_mm:
        ax.plot(dx_mm, dz_mm, 'c+', markersize=12, markeredgewidth=2)
    ax.set_title("Ground Truth\n(Gaussian defect map)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im0, ax=ax, shrink=0.8)
    
    # TFM Reconstruction
    ax = axes[1]
    im1 = ax.imshow(recon_norm, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    ax.set_title("TFM Reconstruction", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    # Overlay
    ax = axes[2]
    im2 = ax.imshow(recon_norm, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    for dx_mm, dz_mm in defects_mm:
        ax.plot(dx_mm, dz_mm, 'c+', markersize=14, markeredgewidth=2,
                label='True defect')
    ax.set_title("Overlay (defects marked)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im2, ax=ax, shrink=0.8)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)
    
    # Metrics text
    metrics_text = (f"PSNR: {metrics['psnr_db']:.2f} dB | "
                    f"SSIM: {metrics['ssim']:.4f} | "
                    f"Pos. Error: {metrics['mean_position_error_mm']:.2f} mm")
    fig.suptitle(f"NDT Ultrasonic TFM Imaging\n{metrics_text}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization to {save_path}")
    
    print("\n" + "=" * 60)
    print("DONE. All results saved to:", results_dir)
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    
    return metrics
