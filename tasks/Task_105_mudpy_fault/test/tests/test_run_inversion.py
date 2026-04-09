import sys
import os
import dill
import numpy as np
import traceback
import json

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ============================================================
# Inject the referee evaluation function verbatim
# ============================================================
def evaluate_results(gt_slip, rec_slip, obs_coords, d_obs, d_pred, patches,
                     fault_length, fault_width, nx_fault, ny_fault):
    """
    Evaluate reconstruction quality and generate visualizations.
    """
    print("[7] Computing metrics ...")

    gt_range = gt_slip.max() - gt_slip.min()
    if gt_range < 1e-15:
        gt_range = 1.0
    psnr = float(peak_signal_noise_ratio(gt_slip, rec_slip, data_range=gt_range))

    data_range = gt_range
    min_side = min(gt_slip.shape)
    win = min(7, min_side)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    ssim_val = float(ssim(gt_slip, rec_slip, data_range=data_range, win_size=win))

    gt_z = gt_slip - gt_slip.mean()
    rec_z = rec_slip - rec_slip.mean()
    denom = np.sqrt(np.sum(gt_z**2) * np.sum(rec_z**2))
    cc = float(np.sum(gt_z * rec_z) / denom) if denom > 1e-15 else 0.0

    rmse = float(np.sqrt(np.mean((gt_slip - rec_slip)**2)))

    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    SSIM = {ssim_val:.4f}")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.4f} m")

    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "CC": float(cc),
        "RMSE": float(rmse),
    }

    print("[8] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_slip)
        np.save(os.path.join(d, "recon_output.npy"), rec_slip)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("[9] Plotting ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.imshow(gt_slip, cmap='hot_r', origin='lower', aspect='auto',
                   extent=[0, fault_length, 0, fault_width])
    ax.set_title("Ground Truth Slip Distribution", fontsize=13)
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")
    plt.colorbar(im, ax=ax, label="Slip (m)")

    ax = axes[0, 1]
    im = ax.imshow(rec_slip, cmap='hot_r', origin='lower', aspect='auto',
                   extent=[0, fault_length, 0, fault_width])
    ax.set_title(f"Reconstructed Slip\nPSNR={metrics['PSNR']:.1f}dB, "
                 f"SSIM={metrics['SSIM']:.3f}, CC={metrics['CC']:.3f}", fontsize=12)
    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")
    plt.colorbar(im, ax=ax, label="Slip (m)")

    ax = axes[1, 0]
    uz_obs = d_obs[2::3]
    uz_pred = d_pred[2::3]
    sc = ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=uz_obs,
                    cmap='RdBu_r', s=30, edgecolors='k', linewidths=0.3)
    ax.set_title("Observed Vertical Displacement", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    plt.colorbar(sc, ax=ax, label="Uz (m)")

    ax = axes[1, 1]
    sc = ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=uz_pred,
                    cmap='RdBu_r', s=30, edgecolors='k', linewidths=0.3,
                    vmin=uz_obs.min(), vmax=uz_obs.max())
    ax.set_title("Predicted Vertical Displacement", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    plt.colorbar(sc, ax=ax, label="Uz (m)")

    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return metrics


# ============================================================
# Main test logic
# ============================================================
def main():
    data_paths = ['/data/yjh/mudpy_fault_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Classify files
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

    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of positional args: {len(args)}")
    print(f"Keyword args keys: {list(kwargs.keys())}")

    if inner_paths:
        # Pattern 2: Chained execution
        print("Detected chained execution pattern (inner data found).")
        # Run outer to get operator
        agent_operator = run_inversion(*args, **kwargs)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        # Execute operator
        agent_result = agent_operator(*inner_args, **inner_kwargs)
    else:
        # Pattern 1: Direct execution
        print("Detected direct execution pattern.")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        std_result = std_output

    # ============================================================
    # Validate outputs
    # ============================================================
    print("\n=== Output Validation ===")

    # Both results should be dicts with 's_hat', 'rec_slip', etc.
    if not isinstance(agent_result, dict):
        print(f"ERROR: agent_result is not a dict, got {type(agent_result)}")
        sys.exit(1)
    if not isinstance(std_result, dict):
        print(f"ERROR: std_result is not a dict, got {type(std_result)}")
        sys.exit(1)

    # Check key fields
    for key in ['s_hat', 'rec_slip']:
        if key not in agent_result:
            print(f"ERROR: Missing key '{key}' in agent result")
            sys.exit(1)
        if key not in std_result:
            print(f"ERROR: Missing key '{key}' in std result")
            sys.exit(1)

    agent_rec_slip = agent_result['rec_slip']
    std_rec_slip = std_result['rec_slip']
    agent_s_hat = agent_result['s_hat']
    std_s_hat = std_result['s_hat']

    print(f"Agent rec_slip shape: {agent_rec_slip.shape}, std rec_slip shape: {std_rec_slip.shape}")
    print(f"Agent s_hat shape: {agent_s_hat.shape}, std s_hat shape: {std_s_hat.shape}")
    print(f"Agent max slip: {agent_rec_slip.max():.4f}, Std max slip: {std_rec_slip.max():.4f}")

    # ============================================================
    # Use evaluate_results to compute metrics
    # We use std_rec_slip as ground truth and compare both agent and std reconstructions
    # ============================================================

    # Extract the inputs used (G, d_obs, nx, ny, lambda_reg)
    # args order: G, d_obs, nx, ny, lambda_reg
    if len(args) >= 5:
        G = args[0]
        d_obs = args[1]
        nx = args[2]
        ny = args[3]
    else:
        G = kwargs.get('G', args[0] if len(args) > 0 else None)
        d_obs = kwargs.get('d_obs', args[1] if len(args) > 1 else None)
        nx = kwargs.get('nx', args[2] if len(args) > 2 else None)
        ny = kwargs.get('ny', args[3] if len(args) > 3 else None)

    # Compute predicted displacements
    d_pred_agent = G @ agent_s_hat
    d_pred_std = G @ std_s_hat

    # For evaluate_results we need obs_coords, patches, fault_length, fault_width
    # These are not directly available from the inversion inputs, so we create placeholders
    n_obs = len(d_obs) // 3

    # Create dummy obs_coords (just indices)
    obs_coords = np.column_stack([
        np.random.uniform(0, 100, n_obs),
        np.random.uniform(0, 100, n_obs)
    ])

    # Dummy patches and fault dimensions
    patches = []
    fault_length = float(nx) * 10.0  # rough estimate
    fault_width = float(ny) * 10.0

    # Use std_rec_slip as ground truth for both evaluations
    gt_slip = std_rec_slip

    print("\n=== Evaluating Agent Result ===")
    try:
        metrics_agent = evaluate_results(
            gt_slip, agent_rec_slip, obs_coords, d_obs, d_pred_agent,
            patches, fault_length, fault_width, nx, ny
        )
    except Exception as e:
        print(f"ERROR in evaluate_results for agent: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Evaluating Standard Result (self-comparison baseline) ===")
    try:
        metrics_std = evaluate_results(
            gt_slip, std_rec_slip, obs_coords, d_obs, d_pred_std,
            patches, fault_length, fault_width, nx, ny
        )
    except Exception as e:
        print(f"ERROR in evaluate_results for std: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Compare metrics
    # ============================================================
    print("\n=== Metric Comparison ===")
    print(f"Agent  -> PSNR: {metrics_agent['PSNR']:.2f}, SSIM: {metrics_agent['SSIM']:.4f}, "
          f"CC: {metrics_agent['CC']:.4f}, RMSE: {metrics_agent['RMSE']:.4f}")
    print(f"Std    -> PSNR: {metrics_std['PSNR']:.2f}, SSIM: {metrics_std['SSIM']:.4f}, "
          f"CC: {metrics_std['CC']:.4f}, RMSE: {metrics_std['RMSE']:.4f}")

    # The standard result compared to itself should give perfect scores
    # Agent result should be very close to standard
    # Primary metrics: PSNR (higher is better), SSIM (higher is better), CC (higher is better), RMSE (lower is better)

    # Direct numerical comparison of the slip vectors
    slip_diff = np.linalg.norm(agent_s_hat - std_s_hat)
    slip_norm = np.linalg.norm(std_s_hat)
    relative_error = slip_diff / (slip_norm + 1e-15)
    print(f"\nDirect comparison:")
    print(f"  ||agent_s - std_s|| = {slip_diff:.6f}")
    print(f"  ||std_s||           = {slip_norm:.6f}")
    print(f"  Relative error      = {relative_error:.6e}")

    # Check if agent result is close enough
    # For a deterministic algorithm like NNLS, results should be nearly identical
    passed = True

    # Check relative error in slip
    if relative_error > 0.05:  # 5% tolerance
        print(f"FAIL: Relative error in slip too large: {relative_error:.4f} > 0.05")
        passed = False
    else:
        print(f"PASS: Relative error in slip acceptable: {relative_error:.6e}")

    # Check PSNR (agent should have high PSNR when compared to std)
    if metrics_agent['PSNR'] < 30.0 and metrics_std['PSNR'] > 100:
        # If std is perfect match (infinite PSNR), agent should have high PSNR
        print(f"WARNING: Agent PSNR is low: {metrics_agent['PSNR']:.2f}")
        if metrics_agent['PSNR'] < 20.0:
            passed = False

    # Check SSIM
    if metrics_agent['SSIM'] < 0.9:
        print(f"FAIL: SSIM too low: {metrics_agent['SSIM']:.4f}")
        passed = False
    else:
        print(f"PASS: SSIM acceptable: {metrics_agent['SSIM']:.4f}")

    # Check CC
    if metrics_agent['CC'] < 0.9:
        print(f"FAIL: CC too low: {metrics_agent['CC']:.4f}")
        passed = False
    else:
        print(f"PASS: CC acceptable: {metrics_agent['CC']:.4f}")

    print(f"\nScores -> Agent PSNR: {metrics_agent['PSNR']:.2f}, Standard PSNR: {metrics_std['PSNR']:.2f}")
    print(f"Scores -> Agent SSIM: {metrics_agent['SSIM']:.4f}, Standard SSIM: {metrics_std['SSIM']:.4f}")

    if passed:
        print("\n=== TEST PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TEST FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)