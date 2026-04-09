import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the Referee (Evaluation Logic) verbatim from Reference B
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def compute_metrics(gt, rec):
    """Compute PSNR, SSIM, and cross-correlation metrics."""
    gt_n = gt / gt.max() if gt.max() > 0 else gt
    rec_n = rec / rec.max() if rec.max() > 0 else rec
    mse = np.mean((gt_n - rec_n) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
    dr = max(gt_n.max() - gt_n.min(), rec_n.max() - rec_n.min())
    if dr < 1e-15:
        dr = 1.0
    ssim_val = ssim(gt_n, rec_n, data_range=dr)
    gz = gt_n - gt_n.mean()
    rz = rec_n - rec_n.mean()
    d = np.sqrt(np.sum(gz ** 2) * np.sum(rz ** 2))
    cc = np.sum(gz * rz) / d if d > 1e-15 else 0.0
    return float(psnr), float(ssim_val), float(cc)


def evaluate_results(data, inversion_result, fov_uas):
    """
    Evaluate reconstruction results and generate outputs.
    
    Computes metrics, saves results, and generates visualization plots.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data
    inversion_result : dict
        Dictionary from run_inversion
    fov_uas : float
        Field of view in micro-arcseconds
        
    Returns
    -------
    dict
        Dictionary containing PSNR, SSIM, and CC metrics
    """
    gt_image = data['gt_image']
    u = data['u']
    v = data['v']
    cleaned = inversion_result['cleaned']
    dirty = inversion_result['dirty']
    
    # Compute metrics
    psnr, ssim_val, cc = compute_metrics(gt_image, cleaned)
    metrics = {"PSNR": float(psnr), "SSIM": float(ssim_val), "CC": float(cc)}
    
    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    SSIM = {ssim_val:.4f}")
    print(f"    CC   = {cc:.4f}")
    
    # Save outputs
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_image)
        np.save(os.path.join(d, "recon_output.npy"), cleaned)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ext = [-fov_uas / 2, fov_uas / 2, -fov_uas / 2, fov_uas / 2]
    
    ax = axes[0, 0]
    im = ax.imshow(gt_image, cmap='afmhot', origin='lower', extent=ext)
    ax.set_title("Ground Truth: Black Hole Shadow", fontsize=13)
    ax.set_xlabel("RA offset (μas)")
    ax.set_ylabel("Dec offset (μas)")
    plt.colorbar(im, ax=ax, label="Flux density")
    
    ax = axes[0, 1]
    ax.scatter(u, v, s=0.3, alpha=0.3, c='navy')
    ax.set_title(f"UV Coverage ({len(u)} points)", fontsize=13)
    ax.set_xlabel("u (cycles/μas)")
    ax.set_ylabel("v (cycles/μas)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    im = ax.imshow(dirty, cmap='afmhot', origin='lower', extent=ext)
    ax.set_title("Dirty Image", fontsize=13)
    ax.set_xlabel("RA offset (μas)")
    ax.set_ylabel("Dec offset (μas)")
    plt.colorbar(im, ax=ax, label="Flux density")
    
    ax = axes[1, 1]
    im = ax.imshow(cleaned, cmap='afmhot', origin='lower', extent=ext)
    ax.set_title(f"CLEAN Reconstruction\nPSNR={metrics['PSNR']:.1f}dB, "
                 f"SSIM={metrics['SSIM']:.3f}, CC={metrics['CC']:.3f}", fontsize=12)
    ax.set_xlabel("RA offset (μas)")
    ax.set_ylabel("Dec offset (μas)")
    plt.colorbar(im, ax=ax, label="Flux density")
    
    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics


# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/ehtim_imaging_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # -----------------------------------------------------------
    # Step 1: Classify files into outer and inner
    # -----------------------------------------------------------
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer (primary) data file found.")
        sys.exit(1)

    # -----------------------------------------------------------
    # Step 2: Load outer data
    # -----------------------------------------------------------
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Outer function: {outer_data.get('func_name', 'unknown')}")
    print(f"  args count: {len(args)}, kwargs keys: {list(kwargs.keys())}")

    # -----------------------------------------------------------
    # Step 3: Run agent's run_inversion
    # -----------------------------------------------------------
    print("\nRunning agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------
    # Step 4: Handle chained vs direct execution
    # -----------------------------------------------------------
    if len(inner_paths) > 0:
        # Chained execution
        print(f"\nChained execution detected. Inner files: {inner_paths}")
        for inner_path in inner_paths:
            print(f"  Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_output = inner_data.get('output', None)

            print(f"  Running agent_output as callable...")
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR calling agent_output: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution
        print("\nDirect execution mode.")
        final_result = agent_output

    # -----------------------------------------------------------
    # Step 5: Evaluate both agent and standard results
    # -----------------------------------------------------------
    # Extract the input data dict (first positional arg) for evaluate_results
    input_data = args[0] if len(args) > 0 else kwargs.get('data', None)
    if input_data is None:
        print("ERROR: Could not extract input data for evaluation.")
        sys.exit(1)

    fov_uas = input_data.get('fov_uas', None)
    if fov_uas is None:
        print("ERROR: fov_uas not found in input data.")
        sys.exit(1)

    # Check that input_data has gt_image for evaluation
    if 'gt_image' not in input_data:
        print("WARNING: gt_image not in input data. Attempting to proceed without full evaluation.")
        # Try a basic comparison instead
        print("Comparing agent output keys vs standard output keys...")
        if isinstance(final_result, dict) and isinstance(std_output, dict):
            for key in std_output:
                if key not in final_result:
                    print(f"  MISSING key in agent output: {key}")
                    sys.exit(1)
            print("  All keys present. Basic check passed.")
            sys.exit(0)
        else:
            print("  Cannot compare non-dict outputs without gt_image.")
            sys.exit(1)

    print("\n=== Evaluating AGENT result ===")
    try:
        metrics_agent = evaluate_results(input_data, final_result, fov_uas)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Evaluating STANDARD result ===")
    try:
        metrics_std = evaluate_results(input_data, std_output, fov_uas)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------
    # Step 6: Compare and report
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    psnr_agent = metrics_agent['PSNR']
    ssim_agent = metrics_agent['SSIM']
    cc_agent = metrics_agent['CC']

    psnr_std = metrics_std['PSNR']
    ssim_std = metrics_std['SSIM']
    cc_std = metrics_std['CC']

    print(f"  Agent  -> PSNR: {psnr_agent:.2f} dB, SSIM: {ssim_agent:.4f}, CC: {cc_agent:.4f}")
    print(f"  Standard -> PSNR: {psnr_std:.2f} dB, SSIM: {ssim_std:.4f}, CC: {cc_std:.4f}")

    # Higher is better for all three metrics.
    # Allow a 10% margin of degradation.
    margin = 0.90  # agent must be at least 90% of standard

    passed = True
    for metric_name, agent_val, std_val in [
        ("PSNR", psnr_agent, psnr_std),
        ("SSIM", ssim_agent, ssim_std),
        ("CC", cc_agent, cc_std),
    ]:
        threshold = std_val * margin
        status = "PASS" if agent_val >= threshold else "FAIL"
        print(f"  {metric_name}: agent={agent_val:.4f}, std={std_val:.4f}, "
              f"threshold(90%)={threshold:.4f} -> {status}")
        if agent_val < threshold:
            passed = False

    print("=" * 60)
    if passed:
        print("OVERALL: PASS - Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL - Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == "__main__":
    main()