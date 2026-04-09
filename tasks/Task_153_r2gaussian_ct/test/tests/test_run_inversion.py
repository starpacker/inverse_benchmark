import sys
import os
import dill
import numpy as np
import traceback

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def evaluate_results(data, result, results_dir=None):
    """
    Compute final metrics, save outputs (metrics JSON, npy arrays, visualization).

    Parameters
    ----------
    data       : dict from load_and_preprocess_data
    result     : dict from run_inversion
    results_dir: str directory to save results (default RESULTS_DIR)

    Returns
    -------
    metrics : dict with PSNR, SSIM, RMSE and other info
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    os.makedirs(results_dir, exist_ok=True)

    phantom = data['phantom']
    recon = result['recon']
    method = result['method']
    psnr = result['psnr']
    ssim_val = result['ssim']
    fbp = result['fbp']
    fp = result['fbp_psnr']
    fs = result['fbp_ssim']
    sp = result['sart_psnr']
    ss = result['sart_ssim']
    gp = result['gs_psnr']
    gss = result['gs_ssim']

    rmse = float(np.sqrt(mean_squared_error(phantom, recon)))
    print(f"\n    BEST: {method} — PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}")

    # Save metrics
    print("[6] Saving...")
    metrics = {
        "task": "r2gaussian_ct",
        "method": method,
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim_val, 4),
        "RMSE": round(rmse, 6),
        "n_angles": data['n_angles'],
        "noise_level": data['noise_level'],
        "image_size": data['size'],
        "n_gaussians": 800,
        "FBP_PSNR": round(fp, 2),
        "FBP_SSIM": round(fs, 4),
        "SART_PSNR": round(sp, 2),
        "SART_SSIM": round(ss, 4),
        "GS_PSNR": round(gp, 2),
        "GS_SSIM": round(gss, 4),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "ground_truth.npy"), phantom)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon)

    # Visualization
    print("[7] Visualization...")
    err = np.abs(phantom - recon)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a in ax:
        a.axis('off')

    n_ang = data['n_angles']

    im0 = ax[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Ground Truth\n(Shepp-Logan)', fontsize=12)
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(fbp, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(f'FBP ({n_ang} angles)\nPSNR={fp:.1f}dB SSIM={fs:.3f}', fontsize=12)
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f'R2-Gaussian CT\nPSNR={psnr:.1f}dB SSIM={ssim_val:.3f}', fontsize=12)
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(err, cmap='hot', vmin=0, vmax=0.3)
    ax[3].set_title(f'Error Map\nRMSE={rmse:.4f}', fontsize=12)
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    plt.suptitle('R2-Gaussian: CT Reconstruction via Gaussian Splatting',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"DONE — {method}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")

    return metrics


# ============================================================
# Main Test Logic
# ============================================================
def main():
    data_paths = ['/data/yjh/r2gaussian_ct_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer and inner data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    print(f"Outer data keys: {list(outer_data.keys())}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")

    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("\n=== Pattern 2: Chained Execution ===")
        print(f"Running run_inversion with outer data to get operator...")
        
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        print(f"Running operator with inner data...")
        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running operator (inner): {e}")
            traceback.print_exc()
            sys.exit(1)

    else:
        # Pattern 1: Direct Execution
        print("\n=== Pattern 1: Direct Execution ===")
        print("Running run_inversion with outer data...")

        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)

        std_result = std_output

    # ============================================================
    # Evaluation Phase
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)

    # We need the input 'data' dict for evaluate_results
    # The first positional argument to run_inversion is 'data'
    input_data = args[0] if len(args) > 0 else kwargs.get('data', None)

    if input_data is None:
        print("ERROR: Could not extract input 'data' dict for evaluation!")
        sys.exit(1)

    # Evaluate Agent result
    print("\n--- Evaluating AGENT result ---")
    try:
        agent_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_agent")
        metrics_agent = evaluate_results(input_data, agent_result, results_dir=agent_results_dir)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate Standard result
    print("\n--- Evaluating STANDARD result ---")
    try:
        std_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_std")
        metrics_std = evaluate_results(input_data, std_result, results_dir=std_results_dir)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Verification & Reporting
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Primary metric: PSNR (higher is better)
    agent_psnr = metrics_agent['PSNR']
    std_psnr = metrics_std['PSNR']

    agent_ssim = metrics_agent['SSIM']
    std_ssim = metrics_std['SSIM']

    agent_rmse = metrics_agent['RMSE']
    std_rmse = metrics_std['RMSE']

    print(f"Agent  -> PSNR: {agent_psnr:.2f}, SSIM: {agent_ssim:.4f}, RMSE: {agent_rmse:.6f}")
    print(f"Standard -> PSNR: {std_psnr:.2f}, SSIM: {std_ssim:.4f}, RMSE: {std_rmse:.6f}")

    print(f"\nScores -> Agent PSNR: {agent_psnr}, Standard PSNR: {std_psnr}")

    # Allow a margin of error: agent should be within 90% of standard PSNR
    # PSNR is "higher is better"
    psnr_threshold = std_psnr * 0.90  # 10% margin

    print(f"PSNR threshold (90% of standard): {psnr_threshold:.2f}")

    if agent_psnr < psnr_threshold:
        print(f"\nFAIL: Agent PSNR ({agent_psnr:.2f}) is significantly below standard ({std_psnr:.2f})")
        print(f"  Agent is below {psnr_threshold:.2f} (90% of standard)")
        sys.exit(1)
    else:
        print(f"\nPASS: Agent PSNR ({agent_psnr:.2f}) is acceptable compared to standard ({std_psnr:.2f})")

    # Also check SSIM as secondary metric
    ssim_threshold = std_ssim * 0.90
    print(f"SSIM threshold (90% of standard): {ssim_threshold:.4f}")

    if agent_ssim < ssim_threshold:
        print(f"WARNING: Agent SSIM ({agent_ssim:.4f}) is below threshold ({ssim_threshold:.4f})")
        print(f"  But PSNR passed, so overall PASS with warning.")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == '__main__':
    main()