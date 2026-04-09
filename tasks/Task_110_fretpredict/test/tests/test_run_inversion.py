import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/fretpredict_sandbox_sandbox')

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter1d


# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

def compute_psnr(gt, recon):
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(gt)
    if peak < 1e-12:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))

def compute_ssim_1d(gt, recon):
    """Compute a 1-D analogue of SSIM."""
    data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-12:
        return 0.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win_sigma = 11.0 / 6.0

    mu_x = gaussian_filter1d(gt, sigma=win_sigma)
    mu_y = gaussian_filter1d(recon, sigma=win_sigma)
    sig_x2 = gaussian_filter1d(gt ** 2, sigma=win_sigma) - mu_x ** 2
    sig_y2 = gaussian_filter1d(recon ** 2, sigma=win_sigma) - mu_y ** 2
    sig_xy = gaussian_filter1d(gt * recon, sigma=win_sigma) - mu_x * mu_y

    sig_x2 = np.maximum(sig_x2, 0)
    sig_y2 = np.maximum(sig_y2, 0)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2))
    return float(np.mean(ssim_map))

def compute_cc(gt, recon):
    g = gt - np.mean(gt)
    r = recon - np.mean(recon)
    denom = np.sqrt(np.sum(g ** 2) * np.sum(r ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(g * r) / denom)

def compute_rmse(gt, recon):
    return float(np.sqrt(np.mean((gt - recon) ** 2)))

def evaluate_results(p_gt, p_recon, r_grid, h, e_edges, params, results_dir, working_dir):
    """
    Compute metrics, save results, and generate visualization.
    """
    R0 = params['R0']
    r_max = params['r_max']
    n_samples = params['n_samples']
    
    # Compute metrics
    psnr_val = compute_psnr(p_gt, p_recon)
    ssim_val = compute_ssim_1d(p_gt, p_recon)
    cc_val = compute_cc(p_gt, p_recon)
    rmse_val = compute_rmse(p_gt, p_recon)

    print(f"\n{'=' * 40}")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  CC:   {cc_val:.4f}")
    print(f"  RMSE: {rmse_val:.6f}")
    print(f"{'=' * 40}")

    metrics = {
        "PSNR": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "CC": round(cc_val, 4),
        "RMSE": round(rmse_val, 6),
    }

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), p_gt)
    np.save(os.path.join(results_dir, "reconstruction.npy"), p_recon)
    np.save(os.path.join(working_dir, "gt_output.npy"), p_gt)
    np.save(os.path.join(working_dir, "recon_output.npy"), p_recon)

    # Visualization
    e_centres = 0.5 * (e_edges[:-1] + e_edges[1:])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: True p(r)
    ax = axes[0, 0]
    ax.fill_between(r_grid, p_gt, alpha=0.4, color='steelblue')
    ax.plot(r_grid, p_gt, 'b-', linewidth=2)
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title("True Distance Distribution p(r)", fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    # Panel 2: FRET efficiency histogram
    ax = axes[0, 1]
    ax.bar(e_centres, h, width=e_centres[1] - e_centres[0],
           color='orange', alpha=0.7, edgecolor='darkorange')
    ax.set_xlabel("FRET Efficiency E", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Observed FRET Efficiency Histogram\n(N={n_samples} molecules)",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Panel 3: Recovered p(r)
    ax = axes[1, 0]
    ax.fill_between(r_grid, p_recon, alpha=0.4, color='tomato')
    ax.plot(r_grid, p_recon, 'r-', linewidth=2)
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title("Recovered Distance Distribution", fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    # Panel 4: Overlay comparison
    ax = axes[1, 1]
    ax.plot(r_grid, p_gt, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(r_grid, p_recon, 'r--', linewidth=2, label='Recovered')
    ax.fill_between(r_grid, p_gt, alpha=0.15, color='blue')
    ax.fill_between(r_grid, p_recon, alpha=0.15, color='red')
    ax.set_xlabel("Distance r (nm)", fontsize=12)
    ax.set_ylabel("p(r)", fontsize=12)
    ax.set_title(f"Overlay Comparison\nPSNR={psnr_val:.1f}dB | SSIM={ssim_val:.4f} | CC={cc_val:.4f}",
                 fontsize=13, fontweight='bold')
    ax.axvline(R0, color='gray', linestyle=':', alpha=0.6, label=f'R0 = {R0} nm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    plt.suptitle("FRET Distance Distribution Recovery (Task 120)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to {results_dir}/")
    
    return metrics


# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    working_dir = '/data/yjh/fretpredict_sandbox_sandbox'
    results_dir = os.path.join(working_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)

    print(f"Outer data: {outer_files}")
    print(f"Inner data: {inner_files}")

    # Load outer data
    assert len(outer_files) == 1, f"Expected exactly 1 outer file, got {len(outer_files)}"
    with open(outer_files[0], 'rb') as f:
        outer_data = dill.load(f)

    print(f"Outer data loaded. Keys: {list(outer_data.keys())}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")

    if len(inner_files) == 0:
        # ============================================================
        # DIRECT EXECUTION MODE
        # ============================================================
        print("\n=== DIRECT EXECUTION MODE ===")

        # Run the agent's function
        agent_output = run_inversion(*args, **kwargs)
        print(f"Agent output type: {type(agent_output)}")

        # ============================================================
        # EVALUATION PHASE
        # ============================================================
        print("\n=== EVALUATION PHASE ===")

        # Extract inputs - handle both args and kwargs
        # The function signature is: run_inversion(E_obs, r_grid, p_gt, R0, n_ebins=80)
        # From the log, we see args is empty and all are in kwargs
        if len(args) >= 3:
            E_obs = args[0]
            r_grid = args[1]
            p_gt = args[2]
            R0 = args[3] if len(args) > 3 else kwargs.get('R0')
        else:
            E_obs = kwargs.get('E_obs')
            r_grid = kwargs.get('r_grid')
            p_gt = kwargs.get('p_gt')
            R0 = kwargs.get('R0')

        assert E_obs is not None, "E_obs not found in inputs"
        assert r_grid is not None, "r_grid not found in inputs"
        assert p_gt is not None, "p_gt not found in inputs"
        assert R0 is not None, "R0 not found in inputs"

        # Extract agent reconstruction
        # Output is a tuple: (p_recon, best_params, h, e_edges)
        if isinstance(agent_output, tuple):
            agent_p_recon = agent_output[0]
            agent_h = agent_output[2]
            agent_e_edges = agent_output[3]
        else:
            raise ValueError(f"Unexpected agent output type: {type(agent_output)}")

        # Extract standard reconstruction
        if isinstance(std_output, tuple):
            std_p_recon = std_output[0]
            std_h = std_output[2]
            std_e_edges = std_output[3]
        else:
            raise ValueError(f"Unexpected std output type: {type(std_output)}")

        # Build params dict for evaluate_results
        r_max = float(np.max(r_grid))
        n_samples = len(E_obs)

        params = {
            'R0': R0,
            'r_max': r_max,
            'n_samples': n_samples,
        }

        # Evaluate agent results
        print("\n--- Agent Evaluation ---")
        agent_results_dir = os.path.join(results_dir, 'agent')
        os.makedirs(agent_results_dir, exist_ok=True)
        agent_working_dir = os.path.join(working_dir, 'agent_work')
        os.makedirs(agent_working_dir, exist_ok=True)
        
        agent_metrics = evaluate_results(
            p_gt, agent_p_recon, r_grid, agent_h, agent_e_edges,
            params, agent_results_dir, agent_working_dir
        )

        # Evaluate standard results
        print("\n--- Standard Evaluation ---")
        std_results_dir = os.path.join(results_dir, 'standard')
        os.makedirs(std_results_dir, exist_ok=True)
        std_working_dir = os.path.join(working_dir, 'std_work')
        os.makedirs(std_working_dir, exist_ok=True)
        
        std_metrics = evaluate_results(
            p_gt, std_p_recon, r_grid, std_h, std_e_edges,
            params, std_results_dir, std_working_dir
        )

        # ============================================================
        # VERIFICATION & REPORTING
        # ============================================================
        print("\n=== VERIFICATION ===")
        
        # Primary metric: PSNR (higher is better)
        agent_psnr = agent_metrics['PSNR']
        std_psnr = std_metrics['PSNR']
        agent_ssim = agent_metrics['SSIM']
        std_ssim = std_metrics['SSIM']
        agent_cc = agent_metrics['CC']
        std_cc = std_metrics['CC']
        agent_rmse = agent_metrics['RMSE']
        std_rmse = std_metrics['RMSE']

        print(f"Scores -> Agent PSNR: {agent_psnr}, Standard PSNR: {std_psnr}")
        print(f"Scores -> Agent SSIM: {agent_ssim}, Standard SSIM: {std_ssim}")
        print(f"Scores -> Agent CC:   {agent_cc}, Standard CC:   {std_cc}")
        print(f"Scores -> Agent RMSE: {agent_rmse}, Standard RMSE: {std_rmse}")

        # Check performance with 10% margin
        # PSNR: higher is better
        psnr_threshold = std_psnr * 0.9
        # SSIM: higher is better
        ssim_threshold = std_ssim * 0.9
        # CC: higher is better
        cc_threshold = std_cc * 0.9
        # RMSE: lower is better
        rmse_threshold = std_rmse * 1.1

        passed = True
        
        if agent_psnr < psnr_threshold:
            print(f"FAIL: Agent PSNR ({agent_psnr:.2f}) < threshold ({psnr_threshold:.2f})")
            passed = False
        else:
            print(f"PASS: Agent PSNR ({agent_psnr:.2f}) >= threshold ({psnr_threshold:.2f})")

        if agent_ssim < ssim_threshold:
            print(f"FAIL: Agent SSIM ({agent_ssim:.4f}) < threshold ({ssim_threshold:.4f})")
            passed = False
        else:
            print(f"PASS: Agent SSIM ({agent_ssim:.4f}) >= threshold ({ssim_threshold:.4f})")

        if agent_cc < cc_threshold:
            print(f"FAIL: Agent CC ({agent_cc:.4f}) < threshold ({cc_threshold:.4f})")
            passed = False
        else:
            print(f"PASS: Agent CC ({agent_cc:.4f}) >= threshold ({cc_threshold:.4f})")

        if agent_rmse > rmse_threshold:
            print(f"FAIL: Agent RMSE ({agent_rmse:.6f}) > threshold ({rmse_threshold:.6f})")
            passed = False
        else:
            print(f"PASS: Agent RMSE ({agent_rmse:.6f}) <= threshold ({rmse_threshold:.6f})")

        if passed:
            print("\n=== ALL CHECKS PASSED ===")
            sys.exit(0)
        else:
            print("\n=== SOME CHECKS FAILED ===")
            sys.exit(1)

    else:
        # ============================================================
        # CHAINED EXECUTION MODE
        # ============================================================
        print("\n=== CHAINED EXECUTION MODE ===")

        # Run outer function to get operator
        operator = run_inversion(*args, **kwargs)
        print(f"Operator type: {type(operator)}")

        # Load inner data
        with open(inner_files[0], 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        inner_std_output = inner_data.get('output', None)

        # Execute operator with inner data
        agent_output = operator(*inner_args, **inner_kwargs)
        std_output = inner_std_output

        print(f"Agent output type: {type(agent_output)}")
        print(f"Std output type: {type(std_output)}")

        # For chained mode, we'd need to figure out how to evaluate
        # This is unlikely for this function but handle gracefully
        print("Chained execution completed. Comparing outputs directly.")
        
        # Direct comparison using PSNR on the outputs
        if isinstance(agent_output, np.ndarray) and isinstance(std_output, np.ndarray):
            psnr = compute_psnr(std_output, agent_output)
            print(f"PSNR between agent and standard output: {psnr:.2f}")
            if psnr > 20.0:
                print("PASS")
                sys.exit(0)
            else:
                print("FAIL")
                sys.exit(1)
        else:
            print("Cannot directly compare non-array outputs in chained mode")
            sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n=== UNHANDLED EXCEPTION ===")
        traceback.print_exc()
        sys.exit(1)