import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_101_pyilc_cmb"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Inject evaluate_results verbatim from Reference B
def evaluate_results(cmb_gt, cmb_rec, data, freqs_ghz, weights, results_dir, assets_dir):
    """
    Compute metrics, save outputs, and generate visualization.
    """
    # Compute metrics
    mse = np.mean((cmb_gt - cmb_rec)**2)
    data_range = cmb_gt.max() - cmb_gt.min()
    psnr = 10.0 * np.log10(data_range**2 / mse) if mse > 0 else 100.0
    ssim_val = ssim(cmb_gt, cmb_rec, data_range=data_range)
    cc = float(np.corrcoef(cmb_gt.ravel(), cmb_rec.ravel())[0, 1])
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "CC": float(cc),
        "RMSE": float(rmse),
    }
    
    # Save outputs
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), cmb_gt)
        np.save(os.path.join(d, "recon_output.npy"), cmb_rec)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    n_pix = cmb_gt.shape[0]
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: selected frequency maps (3 of 6)
    sel = [0, 2, 5]  # 30, 70, 217 GHz
    for idx, si in enumerate(sel):
        ax = fig.add_subplot(3, 3, idx + 1)
        vmax = np.percentile(np.abs(data[si]), 99)
        ax.imshow(data[si], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"{freqs_ghz[si]:.0f} GHz", fontsize=10)
        ax.axis('off')
    
    # Row 2: GT CMB, recovered CMB, residual
    vmax_cmb = np.percentile(np.abs(cmb_gt), 99)
    
    ax = fig.add_subplot(3, 3, 4)
    ax.imshow(cmb_gt, cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title("GT CMB", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(cmb_rec, cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title("Recovered CMB (ILC)", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(3, 3, 6)
    residual = cmb_gt - cmb_rec
    vmax_res = np.percentile(np.abs(residual), 99)
    ax.imshow(residual, cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res)
    ax.set_title(f"Residual (RMS={np.std(residual):.1f} μK)", fontsize=10)
    ax.axis('off')
    
    # Row 3 left: ILC weights
    ax = fig.add_subplot(3, 3, 7)
    ax.bar(range(len(freqs_ghz)), weights, color='steelblue')
    ax.set_xticks(range(len(freqs_ghz)))
    ax.set_xticklabels([f"{f:.0f}" for f in freqs_ghz], fontsize=8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Weight")
    ax.set_title("ILC Weights")
    ax.axhline(0, color='k', lw=0.5)
    
    # Row 3 middle: power spectra comparison
    ax = fig.add_subplot(3, 3, 8)
    ps_gt = np.abs(np.fft.fft2(cmb_gt))**2
    ps_rec = np.abs(np.fft.fft2(cmb_rec))**2
    k = np.arange(1, n_pix // 2)
    kx = np.fft.fftfreq(n_pix, d=1.0) * n_pix
    ky = np.fft.fftfreq(n_pix, d=1.0) * n_pix
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    cl_gt = np.zeros(len(k))
    cl_rec = np.zeros(len(k))
    for i, ki in enumerate(k):
        mask = (K >= ki - 0.5) & (K < ki + 0.5)
        if mask.sum() > 0:
            cl_gt[i] = ps_gt[mask].mean()
            cl_rec[i] = ps_rec[mask].mean()
    
    ax.loglog(k, cl_gt, 'b-', label='GT', lw=1.5)
    ax.loglog(k, cl_rec, 'r--', label='ILC', lw=1.5)
    ax.set_xlabel("Multipole ℓ")
    ax.set_ylabel("C_ℓ")
    ax.set_title("Angular Power Spectrum")
    ax.legend(fontsize=8)
    
    # Row 3 right: metrics text
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    txt = (f"PSNR = {metrics['PSNR']:.2f} dB\n"
           f"SSIM = {metrics['SSIM']:.4f}\n"
           f"CC   = {metrics['CC']:.4f}\n"
           f"RMSE = {metrics['RMSE']:.2f} μK\n"
           f"\nΣ weights = {weights.sum():.6f}")
    ax.text(0.1, 0.5, txt, fontsize=14, family='monospace',
            transform=ax.transAxes, verticalalignment='center')
    ax.set_title("Metrics Summary")
    
    plt.tight_layout()
    for path in [os.path.join(results_dir, "vis_result.png"),
                 os.path.join(assets_dir, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return metrics


def main():
    data_paths = ['/data/yjh/pyilc_cmb_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    
    args = outer_data['args']
    kwargs = outer_data['kwargs']
    std_output = outer_data['output']
    
    # Print input info
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={a.shape}, dtype={a.dtype}")
        else:
            print(f"  arg[{i}]: {type(a).__name__}")
    
    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("Pattern 2: Chained Execution detected.")
        agent_operator = run_inversion(*args, **kwargs)
        
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data['args']
        inner_kwargs = inner_data['kwargs']
        std_result = inner_data['output']
        
        agent_result = agent_operator(*inner_args, **inner_kwargs)
    else:
        # Pattern 1: Direct Execution
        print("Pattern 1: Direct Execution detected.")
        agent_result = run_inversion(*args, **kwargs)
        std_result = std_output
    
    # Extract outputs
    # run_inversion returns (cmb_rec, weights)
    if isinstance(agent_result, tuple):
        agent_cmb_rec, agent_weights = agent_result
    else:
        print("ERROR: Unexpected agent_result type.")
        sys.exit(1)
    
    if isinstance(std_result, tuple):
        std_cmb_rec, std_weights = std_result
    else:
        print("ERROR: Unexpected std_result type.")
        sys.exit(1)
    
    print(f"Agent cmb_rec shape: {agent_cmb_rec.shape}, weights: {agent_weights}")
    print(f"Std cmb_rec shape: {std_cmb_rec.shape}, weights: {std_weights}")
    
    # We need cmb_gt (ground truth CMB) for evaluation.
    # The gen_data_code doesn't explicitly store cmb_gt in the pkl,
    # but we can use std_cmb_rec as the ground truth reference since
    # the standard output IS the reference result.
    # Actually, the evaluate_results expects a separate cmb_gt.
    # The standard result IS our ground truth for comparison.
    cmb_gt = std_cmb_rec
    
    # Extract data and freqs_ghz from the input args
    data = args[0]  # shape (n_freq, n_pix, n_pix)
    freqs_ghz = args[1]  # observation frequencies
    
    # Create separate directories for agent and std evaluation to avoid overwriting
    agent_results_dir = os.path.join(RESULTS_DIR, "agent")
    agent_assets_dir = os.path.join(ASSETS_DIR, "agent")
    std_results_dir = os.path.join(RESULTS_DIR, "std")
    std_assets_dir = os.path.join(ASSETS_DIR, "std")
    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(agent_assets_dir, exist_ok=True)
    os.makedirs(std_results_dir, exist_ok=True)
    os.makedirs(std_assets_dir, exist_ok=True)
    
    # Evaluate agent result against ground truth (std result)
    print("\n--- Evaluating Agent Result ---")
    metrics_agent = evaluate_results(
        cmb_gt, agent_cmb_rec, data, freqs_ghz, agent_weights,
        agent_results_dir, agent_assets_dir
    )
    print(f"Agent Metrics: {metrics_agent}")
    
    # Evaluate std result against itself (should be perfect)
    print("\n--- Evaluating Standard Result ---")
    metrics_std = evaluate_results(
        cmb_gt, std_cmb_rec, data, freqs_ghz, std_weights,
        std_results_dir, std_assets_dir
    )
    print(f"Standard Metrics: {metrics_std}")
    
    # Also save final results to the main directories
    print("\n--- Saving final results to main directories ---")
    final_metrics = evaluate_results(
        cmb_gt, agent_cmb_rec, data, freqs_ghz, agent_weights,
        RESULTS_DIR, ASSETS_DIR
    )
    
    # Compare metrics
    # PSNR, SSIM, CC: Higher is better
    # RMSE: Lower is better
    print("\n=== COMPARISON ===")
    print(f"PSNR  -> Agent: {metrics_agent['PSNR']:.2f}, Standard: {metrics_std['PSNR']:.2f}")
    print(f"SSIM  -> Agent: {metrics_agent['SSIM']:.4f}, Standard: {metrics_std['SSIM']:.4f}")
    print(f"CC    -> Agent: {metrics_agent['CC']:.4f}, Standard: {metrics_std['CC']:.4f}")
    print(f"RMSE  -> Agent: {metrics_agent['RMSE']:.4f}, Standard: {metrics_std['RMSE']:.4f}")
    
    # Direct numerical comparison of outputs
    cmb_diff = np.max(np.abs(agent_cmb_rec - std_cmb_rec))
    weight_diff = np.max(np.abs(agent_weights - std_weights))
    print(f"\nMax abs diff (cmb_rec): {cmb_diff:.2e}")
    print(f"Max abs diff (weights): {weight_diff:.2e}")
    
    # Verification: The agent should produce results very close to standard
    # Since we're comparing agent output to std output as ground truth,
    # PSNR should be very high (ideally infinite if identical)
    # We use a generous threshold
    
    passed = True
    
    # Check if outputs are numerically very close (they should be near-identical
    # since it's the same algorithm)
    if cmb_diff < 1e-6 and weight_diff < 1e-10:
        print("\nOutputs are numerically identical (within floating point tolerance).")
    else:
        # If not identical, check quality metrics
        # PSNR should be very high
        if metrics_agent['PSNR'] < 30.0:  # Very generous threshold
            print(f"WARNING: Agent PSNR ({metrics_agent['PSNR']:.2f}) is below threshold (30.0 dB)")
            passed = False
        
        # SSIM should be very high
        if metrics_agent['SSIM'] < 0.9:
            print(f"WARNING: Agent SSIM ({metrics_agent['SSIM']:.4f}) is below threshold (0.9)")
            passed = False
        
        # CC should be very high
        if metrics_agent['CC'] < 0.9:
            print(f"WARNING: Agent CC ({metrics_agent['CC']:.4f}) is below threshold (0.9)")
            passed = False
    
    # Also verify weights sum to ~1 (ILC constraint)
    weight_sum = agent_weights.sum()
    print(f"\nAgent weights sum: {weight_sum:.6f}")
    if abs(weight_sum - 1.0) > 0.01:
        print(f"WARNING: Weights do not sum to 1 (sum = {weight_sum:.6f})")
        passed = False
    
    if passed:
        print("\n✅ TEST PASSED: Agent run_inversion performs correctly.")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED: Agent run_inversion shows degraded performance.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)