import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# Import the agent's implementation
from agent_run_inversion import run_inversion

# Inject the referee evaluation function verbatim
def evaluate_results(z, F_gt, F_recon, delta_f, results_dir, assets_dir):
    """
    Compute metrics (PSNR, CC, RMSE), save results, and generate visualizations.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Compute metrics
    F_max = np.max(np.abs(F_gt))
    mask = np.abs(F_gt) > 0.01 * F_max
    if np.sum(mask) < 10:
        mask = np.ones(len(z), dtype=bool)

    gt = F_gt[mask]
    rc = F_recon[mask]

    # Scale the reconstruction to match GT (least-squares scaling)
    scale = np.sum(gt * rc) / (np.sum(rc * rc) + 1e-30)
    rc_scaled = rc * scale

    # RMSE
    rmse = np.sqrt(np.mean((gt - rc_scaled)**2))

    # PSNR (relative to signal range)
    signal_range = np.max(gt) - np.min(gt)
    mse = np.mean((gt - rc_scaled)**2)
    psnr = 10 * np.log10(signal_range**2 / (mse + 1e-30))

    # CC (scale-invariant)
    g = gt - np.mean(gt)
    r = rc - np.mean(rc)
    cc = np.sum(g * r) / (np.sqrt(np.sum(g**2) * np.sum(r**2)) + 1e-12)

    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RMSE": float(rmse),
        "scale_factor": float(scale),
    }
    
    F_recon_scaled = F_recon * scale
    
    # Visualization
    z_nm = z * 1e9
    F_gt_nN = F_gt * 1e9
    F_recon_nN = F_recon_scaled * 1e9
    delta_f_Hz = delta_f

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(z_nm, F_gt_nN, "b-", linewidth=2, label="GT Force F(z)")
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Distance z (nm)", fontsize=12)
    axes[0, 0].set_ylabel("Force (nN)", fontsize=12)
    axes[0, 0].set_title("Ground Truth: Lennard-Jones Force", fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_xlim([0, 5])

    axes[0, 1].plot(z_nm, delta_f_Hz, "g-", linewidth=1.5, label="Δf(d)")
    axes[0, 1].set_xlabel("Distance d (nm)", fontsize=12)
    axes[0, 1].set_ylabel("Frequency shift Δf (Hz)", fontsize=12)
    axes[0, 1].set_title("FM-AFM Observable: Frequency Shift", fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_xlim([0, 5])

    axes[1, 0].plot(z_nm, F_gt_nN, "b-", linewidth=2, label="GT Force")
    axes[1, 0].plot(z_nm, F_recon_nN, "r--", linewidth=2, label="Sader-Jarvis Recon")
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].set_xlabel("Distance z (nm)", fontsize=12)
    axes[1, 0].set_ylabel("Force (nN)", fontsize=12)
    axes[1, 0].set_title(
        f"Force Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"CC={metrics['CC']:.4f}",
        fontsize=12,
    )
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].set_xlim([0, 5])

    error_nN = np.abs(F_gt_nN - F_recon_nN)
    axes[1, 1].semilogy(z_nm, error_nN + 1e-15, "m-", linewidth=1.5)
    axes[1, 1].set_xlabel("Distance z (nm)", fontsize=12)
    axes[1, 1].set_ylabel("|Error| (nN)", fontsize=12)
    axes[1, 1].set_title(f"Absolute Error (RMSE={metrics['RMSE']:.2e} N)", fontsize=12)
    axes[1, 1].set_xlim([0, 5])

    plt.tight_layout()
    for p in [os.path.join(results_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), F_gt)
        np.save(os.path.join(d, "recon_output.npy"), F_recon_scaled)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    data_paths = ['/data/yjh/afm_force_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Classify files into outer and inner
    outer_files = []
    inner_files = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Setup directories
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    results_dir_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_std")
    assets_dir_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_std")
    
    try:
        # Load outer data
        assert len(outer_files) >= 1, "No outer data file found!"
        with open(outer_files[0], 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {list(outer_data.keys())}")
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        
        args = outer_data['args']
        kwargs = outer_data['kwargs']
        std_output = outer_data['output']
        
        # Print input info
        print(f"Number of args: {len(args)}")
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray):
                print(f"  arg[{i}]: ndarray shape={a.shape}, dtype={a.dtype}")
            else:
                print(f"  arg[{i}]: {type(a).__name__} = {a}")
        
        if len(inner_files) > 0:
            # Chained execution pattern
            print("\n=== Chained Execution Pattern ===")
            agent_operator = run_inversion(*args, **kwargs)
            
            with open(inner_files[0], 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data['args']
            inner_kwargs = inner_data['kwargs']
            std_result = inner_data['output']
            
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("\n=== Direct Execution Pattern ===")
            agent_result = run_inversion(*args, **kwargs)
            std_result = std_output
        
        print(f"\nAgent result type: {type(agent_result)}")
        print(f"Std result type: {type(std_result)}")
        
        if isinstance(agent_result, np.ndarray):
            print(f"Agent result shape: {agent_result.shape}, range: [{agent_result.min():.6e}, {agent_result.max():.6e}]")
        if isinstance(std_result, np.ndarray):
            print(f"Std result shape: {std_result.shape}, range: [{std_result.min():.6e}, {std_result.max():.6e}]")
        
        # Extract inputs for evaluate_results
        # The function signature is: run_inversion(z, delta_f, k, f0, A)
        # evaluate_results needs: z, F_gt, F_recon, delta_f, results_dir, assets_dir
        # Here F_gt = std_result (ground truth output), F_recon = agent_result
        z = args[0]
        delta_f = args[1]
        
        # Evaluate agent result
        print("\n=== Evaluating Agent Result ===")
        metrics_agent = evaluate_results(
            z=z,
            F_gt=std_result,
            F_recon=agent_result,
            delta_f=delta_f,
            results_dir=results_dir,
            assets_dir=assets_dir
        )
        print(f"Agent Metrics: PSNR={metrics_agent['PSNR']:.4f}, CC={metrics_agent['CC']:.6f}, RMSE={metrics_agent['RMSE']:.6e}, Scale={metrics_agent['scale_factor']:.6f}")
        
        # Evaluate std result against itself (perfect score baseline)
        print("\n=== Evaluating Standard Result (self-comparison baseline) ===")
        metrics_std = evaluate_results(
            z=z,
            F_gt=std_result,
            F_recon=std_result,
            delta_f=delta_f,
            results_dir=results_dir_std,
            assets_dir=assets_dir_std
        )
        print(f"Std Metrics: PSNR={metrics_std['PSNR']:.4f}, CC={metrics_std['CC']:.6f}, RMSE={metrics_std['RMSE']:.6e}, Scale={metrics_std['scale_factor']:.6f}")
        
        # Verification
        print(f"\nScores -> Agent PSNR: {metrics_agent['PSNR']:.4f}, Agent CC: {metrics_agent['CC']:.6f}")
        print(f"Scores -> Std PSNR: {metrics_std['PSNR']:.4f}, Std CC: {metrics_std['CC']:.6f}")
        
        # For the agent vs standard comparison:
        # The agent should produce results very close to the standard.
        # CC should be very high (close to 1.0) since it's the same algorithm.
        # PSNR should be very high since agent should match standard closely.
        
        # Primary check: CC between agent and standard should be very high
        cc_agent = metrics_agent['CC']
        psnr_agent = metrics_agent['PSNR']
        
        # The agent's CC should be near-perfect since it's essentially the same code
        # Allow some margin for numerical differences
        passed = True
        
        if cc_agent < 0.95:
            print(f"FAIL: Agent CC ({cc_agent:.6f}) is too low (threshold: 0.95)")
            passed = False
        else:
            print(f"PASS: Agent CC ({cc_agent:.6f}) >= 0.95")
        
        # PSNR should be reasonably high (agent output should closely match std output)
        if psnr_agent < 20.0:
            print(f"FAIL: Agent PSNR ({psnr_agent:.4f}) is too low (threshold: 20.0 dB)")
            passed = False
        else:
            print(f"PASS: Agent PSNR ({psnr_agent:.4f}) >= 20.0 dB")
        
        # Also do a direct numerical comparison
        if isinstance(agent_result, np.ndarray) and isinstance(std_result, np.ndarray):
            direct_diff = np.max(np.abs(agent_result - std_result))
            rel_diff = direct_diff / (np.max(np.abs(std_result)) + 1e-30)
            print(f"\nDirect comparison: max_abs_diff={direct_diff:.6e}, relative_diff={rel_diff:.6e}")
            
            if rel_diff < 0.01:
                print("PASS: Direct numerical comparison within 1% tolerance")
            elif rel_diff < 0.10:
                print("WARN: Direct numerical comparison within 10% tolerance")
            else:
                print(f"WARN: Direct numerical difference is {rel_diff*100:.2f}%")
        
        if passed:
            print("\n=== TEST PASSED ===")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()