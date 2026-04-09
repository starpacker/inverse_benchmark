import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(B_gt, B_rec, n_lat, n_lon, results_dir, assets_dir, working_dir):
    """
    Compute metrics and generate visualizations.

    Parameters
    ----------
    B_gt : ndarray of shape (n_pix,)
        Ground truth brightness
    B_rec : ndarray of shape (n_pix,)
        Reconstructed brightness
    n_lat : int
        Number of latitude zones
    n_lon : int
        Number of longitude bins
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save website assets
    working_dir : str
        Working directory

    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, CC, RMSE
    """
    # ── Compute PSNR ──
    data_range = B_gt.max() - B_gt.min()
    mse = np.mean((B_gt - B_rec) ** 2)
    if mse < 1e-30:
        psnr_val = 100.0
    elif data_range < 1e-12:
        psnr_val = 0.0
    else:
        psnr_val = 10.0 * np.log10(data_range ** 2 / mse)

    # ── Compute SSIM ──
    gt_f = B_gt.ravel().astype(np.float64)
    rec_f = B_rec.ravel().astype(np.float64)
    drange = gt_f.max() - gt_f.min()
    C1 = (0.01 * drange) ** 2
    C2 = (0.03 * drange) ** 2
    mu_x = gt_f.mean()
    mu_y = rec_f.mean()
    sig_x = gt_f.std()
    sig_y = rec_f.std()
    sig_xy = np.mean((gt_f - mu_x) * (rec_f - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2)
    ssim_val = float(num / den)

    # ── Compute CC ──
    gt_z = gt_f - gt_f.mean()
    rec_z = rec_f - rec_f.mean()
    denom = np.linalg.norm(gt_z) * np.linalg.norm(rec_z)
    if denom < 1e-30:
        cc_val = 0.0
    else:
        cc_val = float(np.dot(gt_z, rec_z) / denom)

    # ── Compute RMSE ──
    rmse_val = float(np.sqrt(np.mean((B_gt - B_rec) ** 2)))

    metrics = {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "CC": cc_val,
        "RMSE": rmse_val,
    }

    # ── Save arrays ──
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "gt_output.npy"), B_gt.reshape(n_lat, n_lon))
    np.save(os.path.join(results_dir, "recon_output.npy"), B_rec.reshape(n_lat, n_lon))
    np.save(os.path.join(assets_dir, "gt_output.npy"), B_gt.reshape(n_lat, n_lon))
    np.save(os.path.join(assets_dir, "recon_output.npy"), B_rec.reshape(n_lat, n_lon))

    # ── Save metrics JSON ──
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(assets_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Generate visualization ──
    B_gt_2d = B_gt.reshape(n_lat, n_lon)
    B_rec_2d = B_rec.reshape(n_lat, n_lon)

    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lon_edges = np.linspace(0, 2 * np.pi, n_lon + 1)
    lats_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lons_1d = 0.5 * (lon_edges[:-1] + lon_edges[1:]) - np.pi

    LON, LAT = np.meshgrid(lons_1d, lats_1d)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             subplot_kw={"projection": "mollweide"})

    im0 = axes[0].pcolormesh(LON, LAT, B_gt_2d, cmap="inferno",
                             vmin=0, vmax=1, shading="auto")
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.05, shrink=0.8)

    im1 = axes[1].pcolormesh(LON, LAT, np.clip(B_rec_2d, 0, 1), cmap="inferno",
                             vmin=0, vmax=1, shading="auto")
    axes[1].set_title("Reconstruction", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.05, shrink=0.8)

    residual = np.abs(B_gt_2d - B_rec_2d)
    im2 = axes[2].pcolormesh(LON, LAT, residual, cmap="hot",
                             vmin=0, vmax=0.5, shading="auto")
    axes[2].set_title("|Residual|", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.05, shrink=0.8)

    fig.suptitle(
        f"Doppler Imaging — Stellar Surface Brightness Recovery\n"
        f"PSNR={metrics['PSNR']:.2f} dB   SSIM={metrics['SSIM']:.4f}   "
        f"CC={metrics['CC']:.4f}",
        fontsize=13, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    vis_paths = [
        os.path.join(results_dir, "vis_result.png"),
        os.path.join(assets_dir, "vis_result.png"),
        os.path.join(working_dir, "vis_result.png"),
    ]
    for p in vis_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)

    return metrics
