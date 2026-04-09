import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

import json

def _match_particles(gt, det, md):
    """Match detected particles to ground truth within distance md."""
    if len(det) == 0 or len(gt) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    D = cdist(gt, det)
    mg = []
    md_ = []
    ug = set()
    ud = set()
    for idx in np.argsort(D, axis=None):
        gi, di = np.unravel_index(idx, D.shape)
        if gi in ug or di in ud:
            continue
        if D[gi, di] > md:
            break
        mg.append(gt[gi])
        md_.append(det[di])
        ug.add(gi)
        ud.add(di)
    return np.array(mg), np.array(md_)

def _rmse(a, b):
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _cc(a, b):
    """Pearson Correlation Coefficient."""
    af, bf = a.ravel(), b.ravel()
    if np.std(af) > 1e-15 and np.std(bf) > 1e-15:
        return float(np.corrcoef(af, bf)[0, 1])
    return 0.0

def _psnr(a, b):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((a - b) ** 2)
    mx = np.max(np.abs(a))
    if mse < 1e-30:
        return 100.0
    if mx < 1e-30:
        return 0.0
    return float(10 * np.log10(mx ** 2 / mse))

def _make_fig(holo, gt, det, mg, md, gv, pixel_size, path):
    """Create visualization figure."""
    um = 1e6
    dx = pixel_size
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ext = [0, holo.shape[0] * dx * um, 0, holo.shape[1] * dx * um]
    im = ax.imshow(holo.T, cmap="gray", origin="lower", extent=ext)
    ax.set_title("Simulated Inline Hologram")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    mip = np.max(gv, axis=0)
    ext2 = [0, mip.shape[0] * dx * um, 0, mip.shape[1] * dx * um]
    im = ax.imshow(mip.T, cmap="hot", origin="lower", extent=ext2)
    ax.set_title("Focus MIP (x-y)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    ax.scatter(gt[:, 0] * um, gt[:, 1] * um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det) > 0:
        ax.scatter(det[:, 0] * um, det[:, 1] * um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg) > 0:
        for g, d in zip(mg * um, md * um):
            ax.plot([g[0], d[0]], [g[1], d[1]], "g--", alpha=0.5, lw=0.8)
    ax.set_title("Top (x-y)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1, 1]
    ax.scatter(gt[:, 0] * um, gt[:, 2] * um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det) > 0:
        ax.scatter(det[:, 0] * um, det[:, 2] * um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg) > 0:
        for g, d in zip(mg * um, md * um):
            ax.plot([g[0], d[0]], [g[2], d[2]], "g--", alpha=0.5, lw=0.8)
    ax.set_title("Side (x-z)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("z (μm)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

def evaluate_results(
    gt_positions,
    detected_positions,
    match_dist,
    hologram,
    gt_particles,
    gradient_volume,
    pixel_size,
    working_dir,
    asset_dir
):
    """
    Evaluate reconstruction quality and save results.
    
    Metrics: RMSE, Pearson CC, PSNR of 3-D positions.
    
    Parameters:
    -----------
    gt_positions : ndarray
        Ground truth positions (N, 3)
    detected_positions : ndarray
        Detected positions (M, 3)
    match_dist : float
        Maximum distance for matching (m)
    hologram : ndarray
        Hologram image for visualization
    gt_particles : ndarray
        Full particle array (N, 4) for visualization
    gradient_volume : ndarray
        3D focus volume for visualization
    pixel_size : float
        Detector pixel pitch (m)
    working_dir : Path
        Directory for saving intermediate results
    asset_dir : Path
        Directory for saving visualization assets
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all computed metrics
    """
    n_gt = len(gt_positions)
    
    # Match detected to ground truth
    mg, md = _match_particles(gt_positions, detected_positions, match_dist)
    nm = len(mg)
    
    # Compute metrics
    if nm > 0:
        r3 = _rmse(mg, md)
        rxy = _rmse(mg[:, :2], md[:, :2])
        rz = _rmse(mg[:, 2:], md[:, 2:])
        cc = _cc(mg, md)
        p = _psnr(mg, md)
    else:
        r3 = rxy = rz = float("inf")
        cc = p = 0.0
    
    print(f"  Matched {nm}/{n_gt}  RMSE={r3 * 1e6:.2f}μm  CC={cc:.4f}  PSNR={p:.2f}dB")
    
    # Save results
    print("\n[5/6] Save …")
    for d in [working_dir, asset_dir]:
        d.mkdir(parents=True, exist_ok=True)
        np.save(str(d / "gt_output.npy"), gt_positions)
        np.save(str(d / "recon_output.npy"), detected_positions)
    
    metrics = dict(
        n_gt=int(n_gt),
        n_detected=int(len(detected_positions)),
        n_matched=int(nm),
        detection_rate=round(nm / n_gt, 4) if n_gt > 0 else 0.0,
        rmse_3d_um=round(r3 * 1e6, 2),
        rmse_xy_um=round(rxy * 1e6, 2),
        rmse_z_um=round(rz * 1e6, 2),
        cc=round(cc, 4),
        psnr_db=round(p, 2)
    )
    
    with open(str(working_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    print("\n[6/6] Plot …")
    for d in [asset_dir, working_dir]:
        _make_fig(
            hologram, gt_particles, detected_positions, mg, md,
            gradient_volume, pixel_size, d / "vis_result.png"
        )
    
    return metrics
