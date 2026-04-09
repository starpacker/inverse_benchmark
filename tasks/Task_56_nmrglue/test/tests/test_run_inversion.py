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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
from scipy.ndimage import label

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the referee function
def evaluate_results(spec_gt, spec_recon, fid_nus, schedule, n_f1, n_peaks, nus_frac, results_dir):
    """
    Compute metrics and generate visualization.
    
    Parameters
    ----------
    spec_gt : np.ndarray
        Ground truth spectrum.
    spec_recon : np.ndarray
        Reconstructed spectrum.
    fid_nus : np.ndarray
        NUS-sampled FID.
    schedule : np.ndarray
        Boolean NUS sampling mask.
    n_f1 : int
        Number of points in indirect dimension.
    n_peaks : int
        Number of peaks in ground truth.
    nus_frac : float
        NUS sampling fraction.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    dict
        Metrics dictionary.
    """
    print("\n[EVAL] Computing metrics ...")
    
    # Normalise both
    gt = spec_gt / np.abs(spec_gt).max()
    rec = spec_recon / np.abs(spec_recon).max()

    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - rec) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # SSIM
    ssim_val = float(ssim_fn(gt, rec, data_range=data_range))

    # CC
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])

    # Relative error
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    # RMSE
    rmse = float(np.sqrt(mse))

    # Peak detection accuracy
    gt_mask = gt > 0.15 * gt.max()
    rec_mask = rec > 0.15 * rec.max()
    gt_labels, n_gt = label(gt_mask)
    rec_labels, n_rec = label(rec_mask)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse,
        "n_peaks_gt": int(n_gt),
        "n_peaks_recon": int(n_rec),
    }

    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save metrics and data
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), spec_recon)
    np.save(os.path.join(results_dir, "ground_truth.npy"), spec_gt)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmax = np.percentile(np.abs(spec_gt), 99)

    # (a) Ground truth spectrum
    ax = axes[0, 0]
    ax.contourf(spec_gt.T, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(a) Ground Truth ({n_peaks} peaks)')
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('F2 [pts]')

    # (b) Reconstructed spectrum
    ax = axes[0, 1]
    ax.contourf(spec_recon.T, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(b) IST Reconstruction (NUS {nus_frac*100:.0f}%)')
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('F2 [pts]')

    # (c) NUS schedule
    ax = axes[1, 0]
    ax.stem(np.where(schedule)[0], np.ones(schedule.sum()),
            linefmt='b-', markerfmt='b.', basefmt='k-')
    ax.set_xlim(0, n_f1)
    ax.set_xlabel('Indirect dimension index')
    ax.set_ylabel('Sampled')
    ax.set_title(f'(c) NUS Schedule ({schedule.sum()}/{n_f1})')

    # (d) 1D slice comparison
    ax = axes[1, 1]
    mid = spec_gt.shape[1] // 2
    ax.plot(spec_gt[:, mid], 'b-', lw=1.5, label='GT', alpha=0.8)
    ax.plot(spec_recon[:, mid], 'r--', lw=1.5, label='IST recon', alpha=0.8)
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('Intensity')
    ax.set_title('(d) 1D Slice (F2 midpoint)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"nmrglue — 2D NMR NUS Reconstruction (IST)\n"
        f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"CC={metrics['CC']:.4f}  |  RE={metrics['RE']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics


def compute_simple_metrics(spec_gt, spec_recon):
    """
    Simplified metric computation without extra parameters.
    Returns PSNR as primary metric.
    """
    # Normalise both
    gt = spec_gt / np.abs(spec_gt).max()
    rec = spec_recon / np.abs(spec_recon).max()

    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - rec) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # SSIM
    ssim_val = float(ssim_fn(gt, rec, data_range=data_range))

    # CC
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])

    # Relative error
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    return {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re
    }


def main():
    # Data paths
    data_paths = ['/data/yjh/nmrglue_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    try:
        # Load outer/primary data
        if not outer_files:
            print("[ERROR] No primary data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"[INFO] Loading primary data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'unknown')
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"[INFO] Function name: {func_name}")
        print(f"[INFO] Args count: {len(args)}")
        print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
        
        # Execute agent function
        print("[INFO] Running agent run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print(f"[INFO] Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
        
        # Check if we have inner data (chained execution)
        if inner_files:
            print("[INFO] Chained execution detected - running inner function...")
            inner_path = inner_files[0]
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            # Execute the returned operator
            final_result = agent_output(*inner_args, **inner_kwargs)
            
        else:
            # Direct execution
            print("[INFO] Direct execution mode...")
            final_result = agent_output
            std_result = std_output
        
        print(f"[INFO] Final result shape: {final_result.shape if hasattr(final_result, 'shape') else type(final_result)}")
        print(f"[INFO] Standard result shape: {std_result.shape if hasattr(std_result, 'shape') else type(std_result)}")
        
        # Compute metrics for both results
        # Since both are spectra (outputs of run_inversion), we compare them
        # by treating std_result as "ground truth" for the agent result
        
        print("\n[EVAL] Evaluating Agent output vs Standard output...")
        
        # Direct comparison metrics
        metrics = compute_simple_metrics(std_result, final_result)
        
        print("\n=== Comparison Metrics ===")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM']:.6f}")
        print(f"  CC:   {metrics['CC']:.6f}")
        print(f"  RE:   {metrics['RE']:.6f}")
        
        # For optimization/reconstruction algorithms, we want:
        # - High PSNR (higher is better)
        # - High SSIM (closer to 1 is better)
        # - High CC (closer to 1 is better)
        # - Low RE (relative error, lower is better)
        
        # Determine success based on reconstruction quality
        # The agent should produce results similar to standard
        
        psnr_threshold = 30.0  # PSNR > 30 dB is generally good
        ssim_threshold = 0.95  # SSIM > 0.95 indicates high similarity
        cc_threshold = 0.95    # CC > 0.95 indicates strong correlation
        re_threshold = 0.1     # RE < 0.1 indicates low relative error
        
        success = True
        failure_reasons = []
        
        if metrics['PSNR'] < psnr_threshold:
            # Be lenient - check if it's at least reasonable
            if metrics['PSNR'] < 20.0:
                failure_reasons.append(f"PSNR too low: {metrics['PSNR']:.2f} < 20.0")
                success = False
        
        if metrics['SSIM'] < ssim_threshold:
            # Be lenient
            if metrics['SSIM'] < 0.8:
                failure_reasons.append(f"SSIM too low: {metrics['SSIM']:.4f} < 0.80")
                success = False
        
        if metrics['CC'] < cc_threshold:
            # Be lenient
            if metrics['CC'] < 0.8:
                failure_reasons.append(f"CC too low: {metrics['CC']:.4f} < 0.80")
                success = False
        
        if metrics['RE'] > re_threshold:
            # Be lenient
            if metrics['RE'] > 0.3:
                failure_reasons.append(f"RE too high: {metrics['RE']:.4f} > 0.30")
                success = False
        
        print("\n=== Test Result ===")
        if success:
            print("[PASS] Agent performance is acceptable!")
            print(f"  PSNR={metrics['PSNR']:.2f}dB, SSIM={metrics['SSIM']:.4f}, CC={metrics['CC']:.4f}, RE={metrics['RE']:.4f}")
            sys.exit(0)
        else:
            print("[FAIL] Agent performance degraded significantly!")
            for reason in failure_reasons:
                print(f"  - {reason}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()