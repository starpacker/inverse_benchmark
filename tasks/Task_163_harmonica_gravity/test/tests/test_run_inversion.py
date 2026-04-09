import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func, peak_signal_noise_ratio as psnr_func
import json

# Inject the referee function (evaluate_results) verbatim from Reference B
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


def main():
    # Data paths
    data_paths = ['/data/yjh/harmonica_gravity_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Parse data paths
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"[INFO] Outer data path: {outer_data_path}")
    print(f"[INFO] Inner data paths: {inner_data_paths}")
    
    # Load outer data
    if outer_data_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from {outer_data_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Outer args count: {len(outer_args)}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Run the agent's function
    try:
        print("[INFO] Running agent's run_inversion...")
        agent_output = run_inversion(*outer_args, **outer_kwargs)
        print("[INFO] Agent's run_inversion completed successfully.")
    except Exception as e:
        print(f"[ERROR] Agent's run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Handle chained execution if inner data exists
    if len(inner_data_paths) > 0:
        print("[INFO] Detected chained execution pattern.")
        # Load inner data
        inner_data_path = inner_data_paths[0]
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"[INFO] Loaded inner data from {inner_data_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        if callable(agent_output):
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("[ERROR] Agent output is not callable but inner data exists.")
            sys.exit(1)
    else:
        # Direct execution
        print("[INFO] Direct execution pattern detected.")
        final_result = agent_output
        std_result = std_output
    
    # Get the input data dictionary (first argument to run_inversion)
    if len(outer_args) > 0:
        input_data = outer_args[0]
    elif 'data' in outer_kwargs:
        input_data = outer_kwargs['data']
    else:
        print("[ERROR] Could not find input data dictionary.")
        sys.exit(1)
    
    # Setup directories for evaluation
    sandbox_dir = "./test_sandbox_output"
    assets_dir = "./test_assets_output"
    sandbox_dir_std = "./test_sandbox_output_std"
    assets_dir_std = "./test_assets_output_std"
    
    # Evaluate agent's result
    try:
        print("\n[INFO] Evaluating agent's result...")
        metrics_agent = evaluate_results(input_data, final_result, sandbox_dir, assets_dir)
        print(f"[INFO] Agent metrics: {metrics_agent}")
    except Exception as e:
        print(f"[ERROR] Failed to evaluate agent's result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    try:
        print("\n[INFO] Evaluating standard result...")
        metrics_std = evaluate_results(input_data, std_result, sandbox_dir_std, assets_dir_std)
        print(f"[INFO] Standard metrics: {metrics_std}")
    except Exception as e:
        print(f"[ERROR] Failed to evaluate standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics for comparison
    # Higher is better: PSNR, SSIM, CC
    # Lower is better: RMSE
    
    psnr_agent = metrics_agent.get('psnr_db', 0)
    psnr_std = metrics_std.get('psnr_db', 0)
    
    ssim_agent = metrics_agent.get('ssim', 0)
    ssim_std = metrics_std.get('ssim', 0)
    
    cc_agent = metrics_agent.get('cc', 0)
    cc_std = metrics_std.get('cc', 0)
    
    rmse_agent = metrics_agent.get('rmse_mgal', float('inf'))
    rmse_std = metrics_std.get('rmse_mgal', float('inf'))
    
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"  PSNR  -> Agent: {psnr_agent:.2f} dB, Standard: {psnr_std:.2f} dB")
    print(f"  SSIM  -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
    print(f"  CC    -> Agent: {cc_agent:.6f}, Standard: {cc_std:.6f}")
    print(f"  RMSE  -> Agent: {rmse_agent:.4f}, Standard: {rmse_std:.4f}")
    print("="*60)
    
    # Verification logic
    # Allow 10% margin for higher-is-better metrics
    # For RMSE (lower is better), allow 10% margin
    
    margin = 0.10  # 10% margin
    
    passed = True
    
    # PSNR check (higher is better)
    if psnr_std > 0:
        if psnr_agent < psnr_std * (1 - margin):
            print(f"[WARN] PSNR degraded significantly: {psnr_agent:.2f} < {psnr_std * (1 - margin):.2f}")
            passed = False
    
    # SSIM check (higher is better)
    if ssim_std > 0:
        if ssim_agent < ssim_std * (1 - margin):
            print(f"[WARN] SSIM degraded significantly: {ssim_agent:.4f} < {ssim_std * (1 - margin):.4f}")
            passed = False
    
    # CC check (higher is better)
    if cc_std > 0:
        if cc_agent < cc_std * (1 - margin):
            print(f"[WARN] CC degraded significantly: {cc_agent:.6f} < {cc_std * (1 - margin):.6f}")
            passed = False
    
    # RMSE check (lower is better)
    if rmse_std > 0:
        if rmse_agent > rmse_std * (1 + margin):
            print(f"[WARN] RMSE degraded significantly: {rmse_agent:.4f} > {rmse_std * (1 + margin):.4f}")
            passed = False
    
    if passed:
        print("\n[SUCCESS] Agent's performance is acceptable.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Agent's performance degraded significantly.")
        sys.exit(1)


if __name__ == "__main__":
    main()