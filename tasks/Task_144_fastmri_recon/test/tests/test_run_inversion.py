import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject Referee: evaluate_results (verbatim from Reference B)
# ============================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def visualize_results(gt, zero_filled, cs_recon, metrics_zf, metrics_cs, save_path):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    gt_disp = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    zf_disp = (zero_filled - zero_filled.min()) / (zero_filled.max() - zero_filled.min() + 1e-12)
    cs_disp = (cs_recon - cs_recon.min()) / (cs_recon.max() - cs_recon.min() + 1e-12)
    error_map = np.abs(gt_disp - cs_disp)

    axes[0].imshow(gt_disp, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(zf_disp, cmap='gray')
    axes[1].set_title(f'Zero-filled\nPSNR={metrics_zf["PSNR"]:.2f} SSIM={metrics_zf["SSIM"]:.3f}',
                      fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(cs_disp, cmap='gray')
    axes[2].set_title(f'CS-TV Recon (ISTA)\nPSNR={metrics_cs["PSNR"]:.2f} SSIM={metrics_cs["SSIM"]:.3f}',
                      fontsize=11)
    axes[2].axis('off')

    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.15)
    axes[3].set_title(f'Error Map (CS)\nRMSE={metrics_cs["RMSE"]:.4f}', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle('FastMRI Reconstruction: Accelerated MRI from Undersampled k-Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")


def evaluate_results(gt_image, recon_cs, recon_zf, params, save_dir=None):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE for both CS and zero-filled reconstructions.
    Saves metrics, visualization, and numpy arrays.
    
    Args:
        gt_image: ground truth image (numpy array)
        recon_cs: CS-TV reconstruction (numpy array)
        recon_zf: zero-filled reconstruction (numpy array)
        params: dictionary with parameters from load_and_preprocess_data
        save_dir: directory to save results (optional)
    
    Returns:
        all_metrics: dictionary with all evaluation metrics
    """
    print("\n[evaluate_results] Computing metrics...")
    
    # Normalize for comparison
    gt_norm = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min() + 1e-12)
    cs_norm = (recon_cs - recon_cs.min()) / (recon_cs.max() - recon_cs.min() + 1e-12)
    zf_norm = (recon_zf - recon_zf.min()) / (recon_zf.max() - recon_zf.min() + 1e-12)
    
    # CS metrics
    psnr_cs = psnr(gt_norm, cs_norm, data_range=1.0)
    ssim_cs = ssim(gt_norm, cs_norm, data_range=1.0)
    rmse_cs = np.sqrt(np.mean((gt_norm - cs_norm) ** 2))
    
    metrics_cs = {
        'PSNR': float(psnr_cs),
        'SSIM': float(ssim_cs),
        'RMSE': float(rmse_cs)
    }
    
    # Zero-filled metrics
    psnr_zf = psnr(gt_norm, zf_norm, data_range=1.0)
    ssim_zf = ssim(gt_norm, zf_norm, data_range=1.0)
    rmse_zf = np.sqrt(np.mean((gt_norm - zf_norm) ** 2))
    
    metrics_zf = {
        'PSNR': float(psnr_zf),
        'SSIM': float(ssim_zf),
        'RMSE': float(rmse_zf)
    }
    
    print(f"  Zero-filled: PSNR={metrics_zf['PSNR']:.2f} dB, "
          f"SSIM={metrics_zf['SSIM']:.4f}, RMSE={metrics_zf['RMSE']:.4f}")
    print(f"  CS-TV Recon: PSNR={metrics_cs['PSNR']:.2f} dB, "
          f"SSIM={metrics_cs['SSIM']:.4f}, RMSE={metrics_cs['RMSE']:.4f}")
    
    # Compile all metrics
    all_metrics = {
        'task': 'fastmri_recon',
        'method': 'ISTA-TV Compressed Sensing MRI Reconstruction',
        'acceleration': params.get('acceleration', 4),
        'image_size': params.get('image_size', 128),
        'PSNR': metrics_cs['PSNR'],
        'SSIM': metrics_cs['SSIM'],
        'RMSE': metrics_cs['RMSE'],
        'zero_filled_PSNR': metrics_zf['PSNR'],
        'zero_filled_SSIM': metrics_zf['SSIM'],
        'zero_filled_RMSE': metrics_zf['RMSE'],
        'sampling_ratio': params.get('sampling_ratio', 0.25),
    }
    
    # Save results if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics JSON
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_path}")
        
        # Save visualization
        vis_path = os.path.join(save_dir, 'reconstruction_result.png')
        visualize_results(gt_image, recon_zf, recon_cs, metrics_zf, metrics_cs, vis_path)
        
        # Save numpy arrays
        np.save(os.path.join(save_dir, 'ground_truth.npy'), gt_image)
        np.save(os.path.join(save_dir, 'reconstruction.npy'), recon_cs)
        print(f"  Saved ground_truth.npy and reconstruction.npy")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CS-TV PSNR:  {metrics_cs['PSNR']:.2f} dB")
    print(f"  CS-TV SSIM:  {metrics_cs['SSIM']:.4f}")
    print(f"  CS-TV RMSE:  {metrics_cs['RMSE']:.4f}")
    print(f"  ZF PSNR:     {metrics_zf['PSNR']:.2f} dB")
    print(f"  ZF SSIM:     {metrics_zf['SSIM']:.4f}")
    
    # Quality check
    if metrics_cs['PSNR'] > 15 and metrics_cs['SSIM'] > 0.5:
        print("\n  ✓ Metrics PASS quality thresholds (PSNR>15, SSIM>0.5)")
    else:
        print("\n  ✗ Metrics BELOW quality thresholds - may need tuning")
    
    print("=" * 60)
    
    return all_metrics


# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/fastmri_recon_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer vs inner data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # Also scan directory for inner data files that may not be listed
    if outer_path:
        data_dir = os.path.dirname(outer_path)
        if os.path.isdir(data_dir):
            for fname in os.listdir(data_dir):
                if fname.startswith('standard_data_parent_function_run_inversion') or \
                   fname.startswith('standard_data_parent_run_inversion'):
                    full = os.path.join(data_dir, fname)
                    if full not in inner_paths and full not in data_paths:
                        inner_paths.append(full)

    print(f"Outer data: {outer_path}")
    print(f"Inner data files: {inner_paths}")

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data: keys={list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    # Determine execution pattern
    is_chained = len(inner_paths) > 0

    if is_chained:
        # Pattern 2: Chained Execution
        print("\n=== Pattern 2: Chained Execution ===")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Load inner data
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data: keys={list(inner_data.keys())}")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running operator (inner): {e}")
            traceback.print_exc()
            sys.exit(1)

    else:
        # Pattern 1: Direct Execution
        print("\n=== Pattern 1: Direct Execution ===")
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
    print("\n=== Evaluation Phase ===")

    # We need a ground truth image and params for evaluate_results.
    # The evaluate_results function needs: gt_image, recon_cs, recon_zf, params
    # 
    # The run_inversion returns {'recon_cs': ..., 'recon_zf': ...}
    # We need to find or reconstruct the ground truth.
    # 
    # Since evaluate_results compares agent vs standard using the same gt,
    # and we don't have the original gt_image in the pkl, we use the
    # standard output's recon_cs as a proxy ground truth to compare quality,
    # OR we compare the agent and standard reconstructions directly.

    # Extract reconstructions
    if isinstance(agent_result, dict):
        agent_recon_cs = agent_result.get('recon_cs', None)
        agent_recon_zf = agent_result.get('recon_zf', None)
    else:
        print("ERROR: agent_result is not a dict")
        sys.exit(1)

    if isinstance(std_result, dict):
        std_recon_cs = std_result.get('recon_cs', None)
        std_recon_zf = std_result.get('recon_zf', None)
    else:
        print("ERROR: std_result is not a dict")
        sys.exit(1)

    # Use the standard recon_cs as the "ground truth" reference for evaluation
    # This tests whether the agent produces equivalent quality output.
    # We evaluate both agent and standard against the same reference (std_recon_cs).
    gt_image = std_recon_cs

    params = {
        'acceleration': 4,
        'image_size': gt_image.shape[0] if gt_image is not None else 128,
        'sampling_ratio': 0.25
    }

    print("\n--- Evaluating Agent Output ---")
    try:
        score_agent = evaluate_results(
            gt_image=gt_image,
            recon_cs=agent_recon_cs,
            recon_zf=agent_recon_zf,
            params=params,
            save_dir=os.path.join(RESULTS_DIR, 'agent')
        )
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Evaluating Standard Output ---")
    try:
        score_std = evaluate_results(
            gt_image=gt_image,
            recon_cs=std_recon_cs,
            recon_zf=std_recon_zf,
            params=params,
            save_dir=os.path.join(RESULTS_DIR, 'standard')
        )
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract primary metrics (PSNR - higher is better, SSIM - higher is better)
    agent_psnr = score_agent['PSNR']
    agent_ssim = score_agent['SSIM']
    agent_rmse = score_agent['RMSE']

    std_psnr = score_std['PSNR']
    std_ssim = score_std['SSIM']
    std_rmse = score_std['RMSE']

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"  Agent PSNR:    {agent_psnr:.2f} dB")
    print(f"  Standard PSNR: {std_psnr:.2f} dB")
    print(f"  Agent SSIM:    {agent_ssim:.4f}")
    print(f"  Standard SSIM: {std_ssim:.4f}")
    print(f"  Agent RMSE:    {agent_rmse:.4f}")
    print(f"  Standard RMSE: {std_rmse:.4f}")

    # Also do a direct comparison: how similar are agent and standard reconstructions?
    # Compute direct PSNR/SSIM between agent_recon_cs and std_recon_cs
    agent_cs_norm = (agent_recon_cs - agent_recon_cs.min()) / (agent_recon_cs.max() - agent_recon_cs.min() + 1e-12)
    std_cs_norm = (std_recon_cs - std_recon_cs.min()) / (std_recon_cs.max() - std_recon_cs.min() + 1e-12)
    
    direct_psnr = psnr(std_cs_norm, agent_cs_norm, data_range=1.0)
    direct_ssim = ssim(std_cs_norm, agent_cs_norm, data_range=1.0)
    direct_rmse = np.sqrt(np.mean((std_cs_norm - agent_cs_norm) ** 2))

    print(f"\n  Direct comparison (Agent vs Standard):")
    print(f"    PSNR: {direct_psnr:.2f} dB")
    print(f"    SSIM: {direct_ssim:.4f}")
    print(f"    RMSE: {direct_rmse:.6f}")

    # Verification
    # The standard output compared against itself should yield perfect scores.
    # The agent should be very close. We use multiple criteria:
    
    # 1. Direct similarity should be very high (PSNR > 25, SSIM > 0.9)
    # 2. Agent quality metrics should not degrade more than 10% from standard
    
    passed = True
    reasons = []

    # Check direct similarity
    if direct_psnr < 25:
        reasons.append(f"Direct PSNR too low: {direct_psnr:.2f} < 25 dB")
        passed = False
    
    if direct_ssim < 0.85:
        reasons.append(f"Direct SSIM too low: {direct_ssim:.4f} < 0.85")
        passed = False

    # Check that agent PSNR is not significantly worse
    # Since std is compared against itself, std_psnr = inf (perfect).
    # So we check agent_psnr is reasonably high (meaning agent ≈ std).
    if agent_psnr < 25:
        reasons.append(f"Agent PSNR against standard too low: {agent_psnr:.2f} < 25 dB")
        passed = False

    if agent_ssim < 0.85:
        reasons.append(f"Agent SSIM against standard too low: {agent_ssim:.4f} < 0.85")
        passed = False

    # Also verify zero-filled reconstructions match
    agent_zf_norm = (agent_recon_zf - agent_recon_zf.min()) / (agent_recon_zf.max() - agent_recon_zf.min() + 1e-12)
    std_zf_norm = (std_recon_zf - std_recon_zf.min()) / (std_recon_zf.max() - std_recon_zf.min() + 1e-12)
    
    zf_psnr = psnr(std_zf_norm, agent_zf_norm, data_range=1.0)
    zf_ssim_val = ssim(std_zf_norm, agent_zf_norm, data_range=1.0)
    print(f"\n  Zero-filled comparison (Agent vs Standard):")
    print(f"    PSNR: {zf_psnr:.2f} dB")
    print(f"    SSIM: {zf_ssim_val:.4f}")

    if zf_psnr < 30:
        reasons.append(f"Zero-filled PSNR mismatch: {zf_psnr:.2f} < 30 dB")
        passed = False

    print(f"\n{'='*60}")
    if passed:
        print("  ✓ TEST PASSED: Agent performance matches standard.")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print("  ✗ TEST FAILED: Agent performance degraded significantly.")
        for r in reasons:
            print(f"    - {r}")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == '__main__':
    main()