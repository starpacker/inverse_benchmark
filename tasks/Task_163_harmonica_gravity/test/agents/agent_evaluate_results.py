import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim_func, peak_signal_noise_ratio as psnr_func

import json

def evaluate_results(
    data: dict,
    inversion_result: dict,
    sandbox_dir: str,
    assets_dir: str
) -> dict:
    """
    Evaluate reconstruction quality and save outputs/visualizations.
    
    This function:
    1. Computes evaluation metrics (PSNR, SSIM, CC, RMSE)
    2. Saves ground truth and reconstructed arrays
    3. Creates a 4-panel visualization
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data
    inversion_result : dict
        Dictionary from run_inversion
    sandbox_dir : str
        Directory for saving sandbox outputs
    assets_dir : str
        Directory for saving asset outputs
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    gt = data['gravity_true_mgal']
    gravity_noisy = data['gravity_noisy']
    recon = inversion_result['gravity_reconstructed']
    coordinates = data['coordinates']
    shape = data['shape']
    unit_label = data['unit_label']
    noise_level = data['noise_level']
    prisms = data['prisms']
    region = data['region']
    
    residual = gt - recon
    
    # RMSE
    rmse = np.sqrt(np.mean(residual**2))
    
    # Correlation Coefficient
    cc = np.corrcoef(gt.ravel(), recon.ravel())[0, 1]
    
    # Normalize both to [0, 1] for PSNR/SSIM computation
    gt_min, gt_max = gt.min(), gt.max()
    data_range = gt_max - gt_min
    
    if data_range > 0:
        gt_norm = (gt - gt_min) / data_range
        recon_norm = (recon - gt_min) / data_range
        recon_norm = np.clip(recon_norm, 0, 1)
    else:
        gt_norm = np.zeros_like(gt)
        recon_norm = np.zeros_like(recon)
    
    # PSNR
    psnr_val = psnr_func(gt_norm, recon_norm, data_range=1.0)
    
    # SSIM
    ssim_val = ssim_func(gt_norm, recon_norm, data_range=1.0)
    
    print(f"\n{'='*60}")
    print(f"  EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"  PSNR  = {psnr_val:.2f} dB")
    print(f"  SSIM  = {ssim_val:.4f}")
    print(f"  CC    = {cc:.6f}")
    print(f"  RMSE  = {rmse:.4f} {unit_label}")
    print(f"{'='*60}\n")
    
    # Save arrays
    np.save(os.path.join(sandbox_dir, "gt_output.npy"), gt)
    np.save(os.path.join(sandbox_dir, "recon_output.npy"), recon)
    np.save(os.path.join(assets_dir, "gt_output.npy"), gt)
    np.save(os.path.join(assets_dir, "recon_output.npy"), recon)
    
    # Save metrics
    metrics = {
        "psnr_db": float(psnr_val),
        "ssim": float(ssim_val),
        "cc": float(cc),
        "rmse_mgal": float(rmse),
        "noise_level_mgal": float(noise_level),
        "n_prisms": len(prisms),
        "grid_shape": list(shape),
        "region_m": list(region),
    }
    
    with open(os.path.join(sandbox_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(assets_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("[INFO] Saved gt_output.npy, recon_output.npy, metrics.json")
    
    # 4-panel visualization
    easting_km = coordinates[0] / 1000.0
    northing_km = coordinates[1] / 1000.0
    extent = [easting_km.min(), easting_km.max(), northing_km.min(), northing_km.max()]
    
    # Shared color limits for first 3 panels
    vmin = min(gt.min(), gravity_noisy.min(), recon.min())
    vmax = max(gt.max(), gravity_noisy.max(), recon.max())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: True gravity anomaly
    ax = axes[0, 0]
    im1 = ax.imshow(gt, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title("(a) True Gravity Anomaly", fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im1, ax=ax, label=f"Gravity ({unit_label})", shrink=0.85)
    
    # Panel 2: Noisy observations
    ax = axes[0, 1]
    im2 = ax.imshow(gravity_noisy, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"(b) Noisy Observations (σ={noise_level} {unit_label})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im2, ax=ax, label=f"Gravity ({unit_label})", shrink=0.85)
    
    # Panel 3: Reconstructed (Equivalent Sources)
    ax = axes[1, 0]
    im3 = ax.imshow(recon, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title("(c) Equivalent Source Reconstruction", fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im3, ax=ax, label=f"Gravity ({unit_label})", shrink=0.85)
    
    # Panel 4: Residual
    ax = axes[1, 1]
    res_abs_max = max(abs(residual.min()), abs(residual.max()))
    im4 = ax.imshow(residual, extent=extent, origin="lower", cmap="RdBu_r",
                    vmin=-res_abs_max, vmax=res_abs_max)
    ax.set_title("(d) Residual (True − Reconstructed)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im4, ax=ax, label=f"Residual ({unit_label})", shrink=0.85)
    
    fig.suptitle(
        f"Gravity Field Inversion via Equivalent Sources\n"
        f"PSNR={psnr_val:.1f} dB | SSIM={ssim_val:.4f} | CC={cc:.4f} | RMSE={rmse:.3f} {unit_label}",
        fontsize=14, fontweight="bold", y=1.02
    )
    
    plt.tight_layout()
    vis_path_sandbox = os.path.join(sandbox_dir, "vis_result.png")
    vis_path_assets = os.path.join(assets_dir, "vis_result.png")
    fig.savefig(vis_path_sandbox, dpi=150, bbox_inches="tight")
    fig.savefig(vis_path_assets, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualization: {vis_path_sandbox}")
    print(f"[INFO] Saved visualization: {vis_path_assets}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  TASK 163: harmonica_gravity — COMPLETE")
    print(f"{'='*60}")
    print(f"  Forward model: {len(prisms)} rectangular prisms")
    print(f"  Grid: {shape[0]}×{shape[1]} @ {data['observation_height']}m elevation")
    print(f"  Inverse: EquivalentSources (depth=5000m, damping=1e-3)")
    print(f"  PSNR  = {psnr_val:.2f} dB")
    print(f"  SSIM  = {ssim_val:.4f}")
    print(f"  CC    = {cc:.6f}")
    print(f"  RMSE  = {rmse:.4f} {unit_label}")
    print(f"{'='*60}")
    
    return metrics
