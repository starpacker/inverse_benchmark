import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the evaluate_results function (Reference B)
def evaluate_results(data_dict, result_dict):
    """
    Evaluate inversion results and save outputs.
    
    Args:
        data_dict: Dictionary containing ground truth data
        result_dict: Dictionary containing inversion results
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    param_names = ["log10_A_gw", "log10_A_red", "gamma_red"]
    true_params = data_dict['true_params']
    medians = result_dict['medians']
    stds = result_dict['stds']
    samples = result_dict['samples']
    freqs = data_dict['freqs']
    
    psd_red_true = data_dict['psd_red_true']
    psd_gw_true = data_dict['psd_gw_true']
    psd_red_recon = result_dict['psd_red_recon']
    psd_gw_recon = result_dict['psd_gw_recon']
    
    # Print parameter recovery results
    print(f"\n  {'Parameter':<15s} {'True':>10s} {'Median':>10s} {'Std':>10s}")
    print("  " + "-" * 50)
    for i, name in enumerate(param_names):
        print(f"  {name:<15s} {true_params[i]:10.3f} {medians[i]:10.3f} {stds[i]:10.3f}")

    # Relative errors
    re_values = {}
    for i, name in enumerate(param_names):
        if abs(true_params[i]) > 1e-10:
            re = abs(medians[i] - true_params[i]) / abs(true_params[i])
        else:
            re = abs(medians[i] - true_params[i])
        re_values[name] = float(re)

    # Cross-correlation of log-PSDs (red noise)
    log_psd_true = np.log10(psd_red_true + 1e-100)
    log_psd_recon = np.log10(psd_red_recon + 1e-100)
    cc_num = np.sum((log_psd_true - log_psd_true.mean()) *
                     (log_psd_recon - log_psd_recon.mean()))
    cc_den = np.sqrt(np.sum((log_psd_true - log_psd_true.mean()) ** 2) *
                     np.sum((log_psd_recon - log_psd_recon.mean()) ** 2))
    psd_cc = float(cc_num / (cc_den + 1e-30))

    # Cross-correlation of log-PSDs (GW)
    log_gw_true = np.log10(psd_gw_true + 1e-100)
    log_gw_recon = np.log10(psd_gw_recon + 1e-100)
    cc_gw_num = np.sum((log_gw_true - log_gw_true.mean()) *
                        (log_gw_recon - log_gw_recon.mean()))
    cc_gw_den = np.sqrt(np.sum((log_gw_true - log_gw_true.mean()) ** 2) *
                        np.sum((log_gw_recon - log_gw_recon.mean()) ** 2))
    psd_gw_cc = float(cc_gw_num / (cc_gw_den + 1e-30))

    mean_re = float(np.mean(list(re_values.values())))

    print(f"\n  Mean relative error: {mean_re:.4f}")
    print(f"  Red noise PSD CC:   {psd_cc:.4f}")
    print(f"  GW PSD CC:          {psd_gw_cc:.4f}")

    # Build metrics dictionary
    metrics = {
        "log10_A_gw_true": float(true_params[0]),
        "log10_A_gw_recovered": float(medians[0]),
        "log10_A_gw_std": float(stds[0]),
        "log10_A_gw_RE": re_values["log10_A_gw"],
        "log10_A_red_true": float(true_params[1]),
        "log10_A_red_recovered": float(medians[1]),
        "log10_A_red_std": float(stds[1]),
        "log10_A_red_RE": re_values["log10_A_red"],
        "gamma_red_true": float(true_params[2]),
        "gamma_red_recovered": float(medians[2]),
        "gamma_red_std": float(stds[2]),
        "gamma_red_RE": re_values["gamma_red"],
        "mean_parameter_RE": mean_re,
        "red_noise_PSD_CC": psd_cc,
        "GW_PSD_CC": psd_gw_cc,
        "n_pulsars": data_dict['n_pulsars'],
        "n_toa": data_dict['n_toa'],
        "n_walkers": result_dict['n_walkers'],
        "n_steps": result_dict['n_steps'],
        "n_burn": result_dict['n_burn'],
    }

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  → metrics.json saved")

    # Save ground truth
    gt_dict = {
        "true_params": true_params,
        "freqs": freqs,
        "psd_red_true": psd_red_true,
        "psd_gw_true": psd_gw_true,
        "residuals": [r for r in data_dict['residuals_all']],
        "hd_matrix": data_dict['hd_matrix'],
    }
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_dict,
            allow_pickle=True)

    # Save reconstruction
    recon_dict = {
        "recovered_params": medians,
        "param_stds": stds,
        "psd_red_recon": psd_red_recon,
        "psd_gw_recon": psd_gw_recon,
        "samples": samples,
    }
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_dict,
            allow_pickle=True)
    print("  → ground_truth.npy, reconstruction.npy saved")

    # ── Visualization ─────────────────────────────────────────────────────
    chain = result_dict['chain']
    n_walkers = result_dict['n_walkers']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0), (0,1), (0,2) Trace plots
    for i, name in enumerate(param_names):
        ax = axes[0, i]
        for w in range(n_walkers):
            ax.plot(chain[:, w, i], alpha=0.3, lw=0.5)
        ax.axhline(true_params[i], color='r', lw=2, label='Truth')
        ax.axhline(medians[i], color='blue', ls='--', lw=1.5, label='Median')
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(f'Trace: {name}')
        ax.legend(fontsize=8)

    # (1,0) Red noise PSD
    ax = axes[1, 0]
    ax.loglog(freqs * 365.25 * 86400, psd_red_true, 'r-', lw=2, label='True red noise')
    ax.loglog(freqs * 365.25 * 86400, psd_red_recon, 'b--', lw=2, label='Recovered')
    ax.set_xlabel('Frequency (1/yr)')
    ax.set_ylabel('PSD (s²/Hz)')
    ax.set_title(f'Red Noise PSD (CC={psd_cc:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) GW PSD
    ax = axes[1, 1]
    ax.loglog(freqs * 365.25 * 86400, psd_gw_true, 'r-', lw=2, label='True GWB')
    ax.loglog(freqs * 365.25 * 86400, psd_gw_recon, 'b--', lw=2, label='Recovered')
    ax.set_xlabel('Frequency (1/yr)')
    ax.set_ylabel('PSD (s²/Hz)')
    ax.set_title(f'GW Background PSD (CC={psd_gw_cc:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Corner-like: 2D posterior (A_gw vs A_red)
    ax = axes[1, 2]
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.2, c='steelblue')
    ax.axvline(true_params[0], color='r', lw=1.5, label='True')
    ax.axhline(true_params[1], color='r', lw=1.5)
    ax.scatter([medians[0]], [medians[1]], c='blue', marker='x', s=100,
               zorder=5, label='Median')
    ax.set_xlabel('log10_A_gw')
    ax.set_ylabel('log10_A_red')
    ax.set_title('Posterior: A_gw vs A_red')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {fig_path} saved")

    print("\n" + "=" * 60)
    print("DONE — PTA Bayesian inference complete")
    print(f"  Mean param RE = {mean_re:.4f}")
    print(f"  Red PSD CC    = {psd_cc:.4f}")
    print(f"  GW PSD CC     = {psd_gw_cc:.4f}")
    print("=" * 60)

    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/enterprise_pta_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent's implementation
        print("\n" + "=" * 60)
        print("Running Agent's run_inversion...")
        print("=" * 60)
        
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if there are inner data files (chained execution)
        if inner_data_files:
            print("\nDetected chained execution pattern...")
            inner_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned callable
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # Extract data_dict from args for evaluation
        # According to the function signature: run_inversion(data_dict, n_walkers, n_steps, n_burn)
        data_dict = args[0] if args else kwargs.get('data_dict')
        
        if data_dict is None:
            print("ERROR: Could not extract data_dict from inputs!")
            sys.exit(1)
        
        # Evaluate agent's results
        print("\n" + "=" * 60)
        print("Evaluating Agent's Results...")
        print("=" * 60)
        
        metrics_agent = evaluate_results(data_dict, final_result)
        
        # Evaluate standard results
        print("\n" + "=" * 60)
        print("Evaluating Standard Results...")
        print("=" * 60)
        
        # For standard results, we need to rename output directory temporarily
        # to avoid overwriting agent results
        metrics_std = None
        if std_result is not None:
            # Create a separate evaluation for standard
            print("\nStandard result keys:", list(std_result.keys()) if isinstance(std_result, dict) else type(std_result))
            
            # Extract key metrics from standard result
            std_medians = std_result.get('medians', None)
            std_stds = std_result.get('stds', None)
            
            if std_medians is not None:
                # Calculate metrics manually for standard
                true_params = data_dict['true_params']
                param_names = ["log10_A_gw", "log10_A_red", "gamma_red"]
                
                re_values_std = {}
                for i, name in enumerate(param_names):
                    if abs(true_params[i]) > 1e-10:
                        re = abs(std_medians[i] - true_params[i]) / abs(true_params[i])
                    else:
                        re = abs(std_medians[i] - true_params[i])
                    re_values_std[name] = float(re)
                
                mean_re_std = float(np.mean(list(re_values_std.values())))
                
                # Calculate PSD correlations for standard
                freqs = data_dict['freqs']
                psd_red_true = data_dict['psd_red_true']
                psd_gw_true = data_dict['psd_gw_true']
                psd_red_recon_std = std_result['psd_red_recon']
                psd_gw_recon_std = std_result['psd_gw_recon']
                
                # Red noise PSD CC
                log_psd_true = np.log10(psd_red_true + 1e-100)
                log_psd_recon_std = np.log10(psd_red_recon_std + 1e-100)
                cc_num = np.sum((log_psd_true - log_psd_true.mean()) *
                                 (log_psd_recon_std - log_psd_recon_std.mean()))
                cc_den = np.sqrt(np.sum((log_psd_true - log_psd_true.mean()) ** 2) *
                                 np.sum((log_psd_recon_std - log_psd_recon_std.mean()) ** 2))
                psd_cc_std = float(cc_num / (cc_den + 1e-30))
                
                # GW PSD CC
                log_gw_true = np.log10(psd_gw_true + 1e-100)
                log_gw_recon_std = np.log10(psd_gw_recon_std + 1e-100)
                cc_gw_num = np.sum((log_gw_true - log_gw_true.mean()) *
                                    (log_gw_recon_std - log_gw_recon_std.mean()))
                cc_gw_den = np.sqrt(np.sum((log_gw_true - log_gw_true.mean()) ** 2) *
                                     np.sum((log_gw_recon_std - log_gw_recon_std.mean()) ** 2))
                psd_gw_cc_std = float(cc_gw_num / (cc_gw_den + 1e-30))
                
                metrics_std = {
                    "mean_parameter_RE": mean_re_std,
                    "red_noise_PSD_CC": psd_cc_std,
                    "GW_PSD_CC": psd_gw_cc_std
                }
                
                print(f"\nStandard Results:")
                print(f"  Mean param RE = {mean_re_std:.4f}")
                print(f"  Red PSD CC    = {psd_cc_std:.4f}")
                print(f"  GW PSD CC     = {psd_gw_cc_std:.4f}")
        
        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        agent_mean_re = metrics_agent['mean_parameter_RE']
        agent_red_cc = metrics_agent['red_noise_PSD_CC']
        agent_gw_cc = metrics_agent['GW_PSD_CC']
        
        print(f"\nAgent Results:")
        print(f"  Mean Parameter RE: {agent_mean_re:.4f}")
        print(f"  Red Noise PSD CC:  {agent_red_cc:.4f}")
        print(f"  GW PSD CC:         {agent_gw_cc:.4f}")
        
        if metrics_std:
            std_mean_re = metrics_std['mean_parameter_RE']
            std_red_cc = metrics_std['red_noise_PSD_CC']
            std_gw_cc = metrics_std['GW_PSD_CC']
            
            print(f"\nStandard Results:")
            print(f"  Mean Parameter RE: {std_mean_re:.4f}")
            print(f"  Red Noise PSD CC:  {std_red_cc:.4f}")
            print(f"  GW PSD CC:         {std_gw_cc:.4f}")
            
            # Determine success
            # For RE: Lower is better
            # For CC: Higher is better (closer to 1.0)
            
            # Allow 20% margin for RE (agent can be up to 20% worse)
            re_ok = agent_mean_re <= std_mean_re * 1.5 + 0.05  # 50% margin + small absolute margin
            
            # For CC, agent should be at least 90% as good
            red_cc_ok = agent_red_cc >= std_red_cc * 0.9 - 0.05
            gw_cc_ok = agent_gw_cc >= std_gw_cc * 0.9 - 0.05
            
            print(f"\nValidation:")
            print(f"  RE check (agent <= std*1.5+0.05): {re_ok} ({agent_mean_re:.4f} vs {std_mean_re * 1.5 + 0.05:.4f})")
            print(f"  Red CC check (agent >= std*0.9-0.05): {red_cc_ok} ({agent_red_cc:.4f} vs {std_red_cc * 0.9 - 0.05:.4f})")
            print(f"  GW CC check (agent >= std*0.9-0.05): {gw_cc_ok} ({agent_gw_cc:.4f} vs {std_gw_cc * 0.9 - 0.05:.4f})")
            
            if re_ok and red_cc_ok and gw_cc_ok:
                print("\n✓ TEST PASSED: Agent performance is acceptable!")
                sys.exit(0)
            else:
                print("\n✗ TEST FAILED: Agent performance degraded significantly!")
                sys.exit(1)
        else:
            # No standard results to compare against
            # Check absolute thresholds
            print("\nNo standard results available for comparison.")
            print("Using absolute thresholds...")
            
            # Reasonable thresholds for MCMC inversion
            re_ok = agent_mean_re < 0.5  # Less than 50% relative error
            red_cc_ok = agent_red_cc > 0.5  # Correlation > 0.5
            gw_cc_ok = agent_gw_cc > 0.5  # Correlation > 0.5
            
            print(f"\nValidation (absolute thresholds):")
            print(f"  RE < 0.5: {re_ok} ({agent_mean_re:.4f})")
            print(f"  Red CC > 0.5: {red_cc_ok} ({agent_red_cc:.4f})")
            print(f"  GW CC > 0.5: {gw_cc_ok} ({agent_gw_cc:.4f})")
            
            if re_ok and red_cc_ok and gw_cc_ok:
                print("\n✓ TEST PASSED: Agent performance meets absolute thresholds!")
                sys.exit(0)
            else:
                print("\n✗ TEST FAILED: Agent performance below absolute thresholds!")
                sys.exit(1)
                
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()