import sys
import os
import dill
import numpy as np
import traceback
import json

# Import the target function from agent module
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Inject the referee evaluation function (from Reference B)
def evaluate_results(dvv_true: np.ndarray, dvv_est: np.ndarray,
                     t: np.ndarray, ccf_ref: np.ndarray,
                     ccf_matrix: np.ndarray, days: np.ndarray,
                     results_dir: str) -> dict:
    """
    Compute dv/v estimation quality metrics and save results.
    
    Parameters:
        dvv_true: true dv/v values
        dvv_est: estimated dv/v values
        t: time axis array
        ccf_ref: reference CCF
        ccf_matrix: matrix of perturbed CCFs
        days: array of day indices
        results_dir: directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Mean absolute error
    mae = float(np.mean(np.abs(dvv_est - dvv_true)))
    
    # Relative error (fraction of amplitude range)
    amp_range = np.max(dvv_true) - np.min(dvv_true)
    rel_error = float(mae / amp_range) if amp_range > 0 else float('inf')
    
    # Correlation coefficient
    cc = float(np.corrcoef(dvv_true, dvv_est)[0, 1])
    
    # PSNR (treating dv/v time series as 1D signal)
    mse = float(np.mean((dvv_est - dvv_true) ** 2))
    peak = float(np.max(np.abs(dvv_true)))
    if mse > 0 and peak > 0:
        psnr = float(20.0 * np.log10(peak / np.sqrt(mse)))
    else:
        psnr = float('inf')
    
    metrics = {
        "dvv_mae": mae,
        "dvv_relative_error": rel_error,
        "dvv_correlation_coefficient": cc,
        "dvv_psnr_dB": psnr,
    }
    
    # Print metrics
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Reference CCF
    ax = axes[0, 0]
    ax.plot(t, ccf_ref, 'k', lw=0.8)
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("(a) Reference CCF")
    ax.set_xlim(t[0], t[-1])
    
    # (b) Reference vs current (one example day)
    ax = axes[0, 1]
    example_day = min(10, len(days) - 1)
    ax.plot(t, ccf_ref, 'k', lw=0.8, label="Reference")
    ax.plot(t, ccf_matrix[example_day], 'r', lw=0.8, alpha=0.7,
            label=f"Current (day {example_day})")
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("(b) Reference vs Perturbed CCF")
    ax.legend(fontsize=9)
    ax.set_xlim(t[0], t[-1])
    
    # (c) True vs estimated dv/v
    ax = axes[1, 0]
    ax.plot(days, dvv_true * 100, 'k-o', ms=3, lw=1.2, label="True dv/v")
    ax.plot(days, dvv_est * 100, 'r-s', ms=3, lw=1.2, label="Estimated dv/v")
    ax.set_xlabel("Day")
    ax.set_ylabel("dv/v (%)")
    ax.set_title("(c) dv/v Time Series")
    ax.legend(fontsize=9)
    
    # (d) Residual
    ax = axes[1, 1]
    residual = (dvv_true - dvv_est) * 100
    ax.bar(days, residual, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("Residual dv/v (%)")
    ax.set_title("(d) Residual (True − Estimated)")
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {save_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), dvv_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), dvv_est)
    print("Arrays saved.")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/seismic_dvv_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Determine execution pattern
    is_chained = len(inner_files) > 0
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent's implementation
        print("\nRunning agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent execution completed.")
        
        if is_chained:
            # Chained execution pattern
            inner_path = inner_files[0]
            print(f"\nLoading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print("Running inner function with agent's output...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Extract results for evaluation
        # The function returns a dict with 'dvv_est' and 'cc_best'
        agent_dvv_est = final_result.get('dvv_est', None)
        std_dvv_est = std_result.get('dvv_est', None)
        
        if agent_dvv_est is None or std_dvv_est is None:
            print("ERROR: Could not extract dvv_est from results!")
            print(f"Agent result keys: {final_result.keys() if isinstance(final_result, dict) else 'not a dict'}")
            print(f"Std result keys: {std_result.keys() if isinstance(std_result, dict) else 'not a dict'}")
            sys.exit(1)
        
        # Extract evaluation parameters from the input args
        # Based on the function signature:
        # run_inversion(ccf_ref, ccf_matrix, t, stretch_range, stretch_steps, coda_tmin)
        ccf_ref = args[0] if len(args) > 0 else kwargs.get('ccf_ref')
        ccf_matrix = args[1] if len(args) > 1 else kwargs.get('ccf_matrix')
        t = args[2] if len(args) > 2 else kwargs.get('t')
        
        # Generate days array based on ccf_matrix shape
        n_days = ccf_matrix.shape[0]
        days = np.arange(n_days)
        
        # For evaluation, we use the standard dvv_est as ground truth
        # and compare agent's estimate against it
        dvv_true = std_dvv_est
        
        # Create results directories
        agent_results_dir = './results_agent'
        std_results_dir = './results_std'
        
        print("\n" + "="*60)
        print("EVALUATING AGENT'S RESULTS")
        print("="*60)
        metrics_agent = evaluate_results(
            dvv_true=dvv_true,
            dvv_est=agent_dvv_est,
            t=t,
            ccf_ref=ccf_ref,
            ccf_matrix=ccf_matrix,
            days=days,
            results_dir=agent_results_dir
        )
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD RESULTS (Self-comparison)")
        print("="*60)
        metrics_std = evaluate_results(
            dvv_true=dvv_true,
            dvv_est=std_dvv_est,
            t=t,
            ccf_ref=ccf_ref,
            ccf_matrix=ccf_matrix,
            days=days,
            results_dir=std_results_dir
        )
        
        # Extract primary metrics for comparison
        # Using correlation coefficient (higher is better) as primary metric
        score_agent_cc = metrics_agent.get('dvv_correlation_coefficient', 0.0)
        score_std_cc = metrics_std.get('dvv_correlation_coefficient', 1.0)
        
        # Also check MAE (lower is better)
        score_agent_mae = metrics_agent.get('dvv_mae', float('inf'))
        score_std_mae = metrics_std.get('dvv_mae', 0.0)
        
        # PSNR (higher is better)
        score_agent_psnr = metrics_agent.get('dvv_psnr_dB', 0.0)
        score_std_psnr = metrics_std.get('dvv_psnr_dB', float('inf'))
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Correlation Coefficient -> Agent: {score_agent_cc:.6f}, Standard: {score_std_cc:.6f}")
        print(f"MAE -> Agent: {score_agent_mae:.6f}, Standard: {score_std_mae:.6f}")
        print(f"PSNR (dB) -> Agent: {score_agent_psnr:.6f}, Standard: {score_std_psnr:.6f}")
        
        # Determine success based on correlation coefficient
        # Allow 10% margin for acceptable performance
        # For correlation, higher is better (max 1.0)
        
        # Check if agent's correlation is at least 90% of standard
        # Since standard is self-comparison (1.0), we check if agent is close enough
        cc_threshold = 0.90  # Agent should have CC >= 0.90
        mae_threshold_multiplier = 2.0  # Agent MAE should not be more than 2x standard
        
        success = True
        failure_reasons = []
        
        if score_agent_cc < cc_threshold:
            success = False
            failure_reasons.append(f"Correlation coefficient {score_agent_cc:.4f} < threshold {cc_threshold}")
        
        # For MAE, if standard is 0 (perfect), agent should be very close
        if score_std_mae > 0:
            if score_agent_mae > score_std_mae * mae_threshold_multiplier:
                success = False
                failure_reasons.append(f"MAE {score_agent_mae:.6f} > {mae_threshold_multiplier}x standard {score_std_mae:.6f}")
        else:
            # If standard MAE is 0, agent should be very small
            if score_agent_mae > 1e-6:
                success = False
                failure_reasons.append(f"MAE {score_agent_mae:.6f} > 1e-6 (standard is 0)")
        
        if success:
            print("\n✓ TEST PASSED: Agent's performance is acceptable.")
            sys.exit(0)
        else:
            print("\n✗ TEST FAILED: Agent's performance degraded significantly.")
            for reason in failure_reasons:
                print(f"  - {reason}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()