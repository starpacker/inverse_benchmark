import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, label
from skimage.metrics import structural_similarity as ssim_fn

# Import target function
from agent_run_inversion import run_inversion

# Setup paths
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# INJECTED REFEREE CODE (from Reference B)
# ============================================================================

def compute_metrics(gt, rec):
    """Compute SAR image quality metrics."""
    # Normalise both to [0, 1]
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}

def evaluate_results(sigma_gt, inversion_results, results_dir):
    """
    Evaluate SAR reconstruction results and generate visualizations.
    
    Computes quality metrics (PSNR, SSIM, CC, RE, RMSE) and applies
    post-processing to optimize reconstruction quality.
    
    Args:
        sigma_gt: ground truth scene reflectivity
        inversion_results: dict containing img_bp and img_pfa
        results_dir: directory to save results
        
    Returns:
        dict containing final metrics and best reconstruction
    """
    img_bp = inversion_results['img_bp']
    img_pfa = inversion_results['img_pfa']
    
    # Compute metrics for both methods
    m_bp = compute_metrics(sigma_gt, img_bp)
    m_pfa = compute_metrics(sigma_gt, img_pfa)
    print(f"  Backprojection CC={m_bp['CC']:.4f}")
    print(f"  PFA CC={m_pfa['CC']:.4f}")
    
    # Choose best method
    if m_bp['CC'] >= m_pfa['CC']:
        img_rec = img_bp
        metrics = m_bp
        method = "Backprojection"
    else:
        img_rec = img_pfa
        metrics = m_pfa
        method = "PFA"
    print(f"\n  → Using {method} (higher CC)")
    
    # ── Normalize and clean reconstruction ──
    img_rec = img_rec / max(img_rec.max(), 1e-12)
    sigma_gt_norm = sigma_gt / max(sigma_gt.max(), 1e-12)

    # --- Post-processing: median filter + threshold + Gaussian blur ---
    img_med = median_filter(img_rec, size=9)
    img_med[img_med < 0.16] = 0
    img_med = gaussian_filter(img_med, sigma=0.7)
    img_med = img_med / max(img_med.max(), 1e-12)

    # Also try: simple thresholded version
    img_thresh = img_rec.copy()
    img_thresh[img_thresh < 0.20] = 0
    img_thresh = gaussian_filter(img_thresh, sigma=1.0)
    img_thresh = img_thresh / max(img_thresh.max(), 1e-12)

    # Compare approaches
    m_med = compute_metrics(sigma_gt_norm, img_med)
    m_thresh = compute_metrics(sigma_gt_norm, img_thresh)
    m_raw = compute_metrics(sigma_gt_norm, img_rec)

    print(f"\n  Raw normalized:     CC={m_raw['CC']:.4f}, PSNR={m_raw['PSNR']:.2f}")
    print(f"  Median+thresh+blur: CC={m_med['CC']:.4f}, PSNR={m_med['PSNR']:.2f}")
    print(f"  Thresholded+blur:   CC={m_thresh['CC']:.4f}, PSNR={m_thresh['PSNR']:.2f}")

    # Pick the best approach by CC
    candidates = [
        (img_rec, m_raw, "raw"),
        (img_thresh, m_thresh, "thresholded"),
        (img_med, m_med, "median-filtered"),
    ]
    best_img, best_metrics, best_name = max(candidates, key=lambda x: x[1]['CC'])
    print(f"  → Best approach: {best_name}")
    img_rec_final = best_img
    metrics_final = best_metrics

    print(f"\n  Final: CC={metrics_final['CC']:.4f}, PSNR={metrics_final['PSNR']:.2f}, SSIM={metrics_final['SSIM']:.4f}")

    # Print evaluation metrics
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics_final.items()):
        print(f"  {k:20s} = {v}")

    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_final, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), img_rec_final)
    np.save(os.path.join(results_dir, "ground_truth.npy"), sigma_gt_norm)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gt_db = 20 * np.log10(sigma_gt_norm / max(sigma_gt_norm.max(), 1e-12) + 1e-6)
    bp_db = 20 * np.log10(img_bp / max(img_bp.max(), 1e-12) + 1e-6)
    pfa_db = 20 * np.log10(img_pfa / max(img_pfa.max(), 1e-12) + 1e-6)

    vmin = -40
    for ax, img, title in zip(axes,
                               [gt_db, bp_db, pfa_db],
                               ['Ground Truth', 'Backprojection', 'PFA']):
        im = ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=0,
                        origin='lower', aspect='auto')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='dB')

    fig.suptitle(
        f"RITSAR — SAR Image Formation\n"
        f"PSNR={metrics_final['PSNR']:.1f} dB | CC={metrics_final['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'metrics': metrics_final,
        'img_rec': img_rec_final,
        'sigma_gt_norm': sigma_gt_norm,
        'img_bp': img_bp,
        'img_pfa': img_pfa
    }

# ============================================================================
# TEST LOGIC
# ============================================================================

def main():
    # Data paths provided
    data_paths = ['/data/yjh/RITSAR_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 60)
    print("QA Test for run_inversion")
    print("=" * 60)
    
    # Analyze data paths to determine execution pattern
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"\nOuter data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"\nLoaded outer data: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"\nInput args count: {len(args)}")
    print(f"Input kwargs keys: {list(kwargs.keys())}")
    
    # Execute agent function
    try:
        print("\n" + "-" * 40)
        print("Running Agent's run_inversion...")
        print("-" * 40)
        agent_output = run_inversion(*args, **kwargs)
        print("\nAgent execution completed successfully.")
    except Exception as e:
        print(f"ERROR during agent execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have chained execution (inner data)
    if inner_data_paths:
        # Pattern 2: Chained execution
        print("\n" + "-" * 40)
        print("Chained execution detected - running inner function...")
        print("-" * 40)
        
        for inner_path in inner_data_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_inner_output = inner_data.get('output', None)
                
                # Execute the operator returned by run_inversion
                if callable(agent_output):
                    final_agent_result = agent_output(*inner_args, **inner_kwargs)
                    std_result = std_inner_output
                else:
                    final_agent_result = agent_output
                    std_result = std_output
                    
            except Exception as e:
                print(f"ERROR loading/executing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Pattern 1: Direct execution
        final_agent_result = agent_output
        std_result = std_output
    
    print("\n" + "=" * 60)
    print("Evaluating Results")
    print("=" * 60)
    
    # We need sigma_gt (ground truth) for evaluation
    # The evaluate_results function expects (sigma_gt, inversion_results, results_dir)
    # We need to extract sigma_gt from somewhere - it should be in the test context
    
    # Check if sigma_gt is available in the data
    # Looking at the function signature, sigma_gt should be passed separately
    # For this test, we'll compare the outputs directly
    
    # The agent output and std output should both be dicts with 'img_bp' and 'img_pfa'
    if isinstance(final_agent_result, dict) and isinstance(std_result, dict):
        print("\nComparing reconstruction outputs...")
        
        # Extract images
        agent_img_bp = final_agent_result.get('img_bp')
        agent_img_pfa = final_agent_result.get('img_pfa')
        std_img_bp = std_result.get('img_bp')
        std_img_pfa = std_result.get('img_pfa')
        
        if agent_img_bp is not None and std_img_bp is not None:
            # Compute metrics comparing agent to standard
            bp_metrics = compute_metrics(std_img_bp, agent_img_bp)
            print(f"\nBackprojection comparison:")
            print(f"  PSNR: {bp_metrics['PSNR']:.2f} dB")
            print(f"  SSIM: {bp_metrics['SSIM']:.4f}")
            print(f"  CC: {bp_metrics['CC']:.4f}")
            
        if agent_img_pfa is not None and std_img_pfa is not None:
            pfa_metrics = compute_metrics(std_img_pfa, agent_img_pfa)
            print(f"\nPFA comparison:")
            print(f"  PSNR: {pfa_metrics['PSNR']:.2f} dB")
            print(f"  SSIM: {pfa_metrics['SSIM']:.4f}")
            print(f"  CC: {pfa_metrics['CC']:.4f}")
        
        # Use CC as primary metric for success determination
        # Higher CC means better correlation with standard
        agent_bp_quality = bp_metrics['CC'] if 'bp_metrics' in dir() else 0
        agent_pfa_quality = pfa_metrics['CC'] if 'pfa_metrics' in dir() else 0
        
        # Both metrics should show high correlation (> 0.9 means very similar results)
        threshold = 0.9  # Allow 10% margin
        
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        bp_pass = agent_bp_quality >= threshold
        pfa_pass = agent_pfa_quality >= threshold
        
        print(f"\nBackprojection CC: {agent_bp_quality:.4f} (threshold: {threshold}) -> {'PASS' if bp_pass else 'FAIL'}")
        print(f"PFA CC: {agent_pfa_quality:.4f} (threshold: {threshold}) -> {'PASS' if pfa_pass else 'FAIL'}")
        
        # Also check that the shapes match
        shape_match_bp = agent_img_bp.shape == std_img_bp.shape if (agent_img_bp is not None and std_img_bp is not None) else False
        shape_match_pfa = agent_img_pfa.shape == std_img_pfa.shape if (agent_img_pfa is not None and std_img_pfa is not None) else False
        
        print(f"\nShape match BP: {shape_match_bp}")
        print(f"Shape match PFA: {shape_match_pfa}")
        
        # Final determination
        if bp_pass and pfa_pass and shape_match_bp and shape_match_pfa:
            print("\n✓ TEST PASSED: Agent output matches standard within acceptable tolerance.")
            sys.exit(0)
        else:
            # Check if at least one method passes with high quality
            if (bp_pass or pfa_pass) and (shape_match_bp and shape_match_pfa):
                print("\n✓ TEST PASSED: At least one reconstruction method matches standard.")
                sys.exit(0)
            else:
                print("\n✗ TEST FAILED: Agent output does not match standard.")
                sys.exit(1)
    else:
        # Fallback: basic type and shape comparison
        print("\nFallback comparison (non-dict outputs)...")
        
        if type(final_agent_result) != type(std_result):
            print(f"Type mismatch: agent={type(final_agent_result)}, std={type(std_result)}")
            sys.exit(1)
        
        if hasattr(final_agent_result, 'shape') and hasattr(std_result, 'shape'):
            if final_agent_result.shape != std_result.shape:
                print(f"Shape mismatch: agent={final_agent_result.shape}, std={std_result.shape}")
                sys.exit(1)
            
            # Compute correlation
            cc = np.corrcoef(final_agent_result.ravel(), std_result.ravel())[0, 1]
            print(f"Correlation coefficient: {cc:.4f}")
            
            if cc >= 0.9:
                print("\n✓ TEST PASSED")
                sys.exit(0)
            else:
                print("\n✗ TEST FAILED")
                sys.exit(1)
        
        print("\n✓ TEST PASSED (basic type check)")
        sys.exit(0)


if __name__ == "__main__":
    main()