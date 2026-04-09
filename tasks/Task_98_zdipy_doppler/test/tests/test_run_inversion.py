import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the agent's implementation
from agent_run_inversion import run_inversion


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


def main():
    data_paths = ['/data/yjh/zdipy_doppler_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer vs inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Function: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}, kwargs keys: {list(kwargs.keys())}")

    # Run the agent's implementation
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Chained execution pattern
        print("Chained execution detected. Running inner calls...")
        for ip in inner_paths:
            print(f"Loading inner data from: {ip}")
            with open(ip, 'rb') as f:
                inner_data = dill.load(f)
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR running inner call: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution pattern
        final_result = agent_output
        std_result = std_output

    # Extract B_rec from results (run_inversion returns (B_rec, lam_used))
    if isinstance(final_result, tuple):
        agent_B_rec = final_result[0]
        agent_lam = final_result[1]
    else:
        agent_B_rec = final_result
        agent_lam = None

    if isinstance(std_result, tuple):
        std_B_rec = std_result[0]
        std_lam = std_result[1]
    else:
        std_B_rec = std_result
        std_lam = None

    print(f"Agent B_rec shape: {agent_B_rec.shape}, std B_rec shape: {std_B_rec.shape}")
    if agent_lam is not None and std_lam is not None:
        print(f"Agent lambda: {agent_lam}, Std lambda: {std_lam}")

    # We need B_gt and grid dimensions for evaluate_results
    # Since evaluate_results needs B_gt, we use std_B_rec as the ground truth reference
    # (the standard output IS the expected reconstruction)
    # For the grid dimensions, we infer from the pixel count
    n_pix = agent_B_rec.shape[0]

    # Common grid sizes for Doppler imaging
    # Try to find reasonable n_lat, n_lon that multiply to n_pix
    n_lat = None
    n_lon = None
    for candidate_lat in [10, 12, 15, 18, 20, 24, 30, 36]:
        if n_pix % candidate_lat == 0:
            candidate_lon = n_pix // candidate_lat
            if candidate_lon >= candidate_lat:
                n_lat = candidate_lat
                n_lon = candidate_lon

    if n_lat is None:
        # Fallback: try sqrt-ish factorization
        sq = int(np.sqrt(n_pix))
        for i in range(sq, 0, -1):
            if n_pix % i == 0:
                n_lat = i
                n_lon = n_pix // i
                break

    print(f"Grid: n_lat={n_lat}, n_lon={n_lon}, n_pix={n_pix}")

    # Set up directories
    working_dir = os.path.abspath("./test_working")
    results_dir = os.path.join(working_dir, "results")
    assets_dir = os.path.join(working_dir, "assets")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # Use std_B_rec as ground truth for evaluation
    B_gt = std_B_rec

    # Evaluate agent output
    print("\n=== Evaluating Agent Output ===")
    try:
        agent_results_dir = os.path.join(results_dir, "agent")
        agent_assets_dir = os.path.join(assets_dir, "agent")
        metrics_agent = evaluate_results(B_gt, agent_B_rec, n_lat, n_lon,
                                         agent_results_dir, agent_assets_dir, working_dir)
        print(f"Agent Metrics: {metrics_agent}")
    except Exception as e:
        print(f"ERROR evaluating agent: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard output (should be perfect since B_gt == std_B_rec)
    print("\n=== Evaluating Standard Output ===")
    try:
        std_results_dir = os.path.join(results_dir, "std")
        std_assets_dir = os.path.join(assets_dir, "std")
        metrics_std = evaluate_results(B_gt, std_B_rec, n_lat, n_lon,
                                       std_results_dir, std_assets_dir, working_dir)
        print(f"Standard Metrics: {metrics_std}")
    except Exception as e:
        print(f"ERROR evaluating standard: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Compare results
    print("\n=== Comparison ===")
    print(f"Scores -> Agent PSNR: {metrics_agent['PSNR']:.4f}, Standard PSNR: {metrics_std['PSNR']:.4f}")
    print(f"Scores -> Agent SSIM: {metrics_agent['SSIM']:.4f}, Standard SSIM: {metrics_std['SSIM']:.4f}")
    print(f"Scores -> Agent CC: {metrics_agent['CC']:.4f}, Standard CC: {metrics_std['CC']:.4f}")
    print(f"Scores -> Agent RMSE: {metrics_agent['RMSE']:.6f}, Standard RMSE: {metrics_std['RMSE']:.6f}")

    # Since std is the ground truth itself, agent metrics should show near-perfect scores
    # PSNR should be very high (ideally 100 or close), SSIM ~1, CC ~1, RMSE ~0

    # Direct numerical comparison of outputs
    max_diff = np.max(np.abs(agent_B_rec - std_B_rec))
    mean_diff = np.mean(np.abs(agent_B_rec - std_B_rec))
    print(f"\nDirect comparison: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")

    # Check lambda values
    if agent_lam is not None and std_lam is not None:
        lam_diff = abs(agent_lam - std_lam)
        print(f"Lambda difference: {lam_diff:.8f}")
        if lam_diff > abs(std_lam) * 0.01:  # 1% tolerance
            print(f"WARNING: Lambda values differ significantly: agent={agent_lam}, std={std_lam}")

    # Determine pass/fail
    passed = True

    # Check PSNR (higher is better) - agent should achieve very high PSNR
    if metrics_agent['PSNR'] < 60.0:  # Very generous threshold
        print(f"FAIL: Agent PSNR {metrics_agent['PSNR']:.4f} is too low (< 60 dB)")
        passed = False

    # Check RMSE (lower is better) - should be near zero
    if metrics_agent['RMSE'] > 0.01:
        print(f"FAIL: Agent RMSE {metrics_agent['RMSE']:.6f} is too high (> 0.01)")
        passed = False

    # Check direct numerical agreement
    if max_diff > 1e-6:
        print(f"WARNING: Max difference {max_diff:.8f} > 1e-6")
        # Still allow if metrics are good
        if metrics_agent['PSNR'] < 40.0:
            print(f"FAIL: Numerical difference too large and PSNR too low")
            passed = False

    if passed:
        print("\n✓ TEST PASSED: Agent output matches standard within acceptable tolerance.")
        sys.exit(0)
    else:
        print("\n✗ TEST FAILED: Agent output does not match standard.")
        sys.exit(1)


if __name__ == "__main__":
    main()