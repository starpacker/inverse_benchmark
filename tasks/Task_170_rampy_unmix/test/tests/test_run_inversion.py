import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# INJECTED REFEREE FUNCTION (from Reference B)
# ============================================================

def evaluate_results(data, result, output_dir='results'):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data containing ground truth.
    result : dict
        Dictionary from run_inversion containing reconstruction results.
    output_dir : str
        Directory to save outputs.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    wavenumber = data['wavenumber']
    pure_components = data['pure_components']
    true_weights = data['true_weights']
    clean_spectra = data['clean_spectra']
    baselines = data['baselines']
    observed_spectra = data['observed_spectra']
    n_components = data['n_components']
    
    corrected_spectra = result['corrected_spectra']
    fitted_baselines = result['fitted_baselines']
    H_matched = result['recovered_components']
    W_matched = result['recovered_weights']
    col_ind = result['permutation']
    
    n_mixtures = observed_spectra.shape[0]
    
    # ============================================================
    # EVALUATION METRICS
    # ============================================================
    
    metrics = {}
    
    # --- (a) Baseline-corrected PSNR ---
    # Compare corrected spectra vs clean spectra (ground truth without baseline/noise)
    mse_baseline = np.mean((corrected_spectra - clean_spectra) ** 2)
    max_val = clean_spectra.max()
    psnr_baseline = 10 * np.log10(max_val ** 2 / mse_baseline)
    metrics['baseline_corrected_PSNR_dB'] = float(round(psnr_baseline, 2))
    
    # Correlation between corrected and clean
    cc_baseline_list = []
    for i in range(n_mixtures):
        cc = np.corrcoef(corrected_spectra[i], clean_spectra[i])[0, 1]
        cc_baseline_list.append(cc)
    metrics['baseline_corrected_mean_CC'] = float(round(np.mean(cc_baseline_list), 4))
    
    # --- (b) Component spectrum correlation ---
    comp_cc_list = []
    for i in range(n_components):
        cc = np.corrcoef(pure_components[i], H_matched[i])[0, 1]
        comp_cc_list.append(cc)
        metrics[f'component_{i+1}_CC'] = float(round(cc, 4))
    metrics['mean_component_CC'] = float(round(np.mean(comp_cc_list), 4))
    
    # --- (c) Mixing proportion relative error ---
    re_per_sample = np.abs(W_matched - true_weights) / (np.abs(true_weights) + 1e-10)
    mean_re = np.mean(re_per_sample)
    metrics['mixing_proportion_mean_RE'] = float(round(mean_re, 4))
    
    # Mixing proportion correlation (per component)
    for i in range(n_components):
        cc_w = np.corrcoef(true_weights[:, i], W_matched[:, i])[0, 1]
        metrics[f'weight_component_{i+1}_CC'] = float(round(cc_w, 4))
    
    # Overall PSNR (as a summary metric)
    metrics['PSNR'] = metrics['baseline_corrected_PSNR_dB']
    
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # ============================================================
    # SAVE GROUND TRUTH AND RECONSTRUCTION
    # ============================================================
    
    # Ground truth: dict with all true data
    gt_data = {
        'wavenumber': wavenumber,
        'pure_components': pure_components,
        'true_weights': true_weights,
        'clean_spectra': clean_spectra,
        'baselines': baselines,
        'observed_spectra': observed_spectra,
    }
    np.save(os.path.join(output_dir, 'ground_truth.npy'), gt_data, allow_pickle=True)
    
    # Reconstruction: dict with all recovered data
    recon_data = {
        'corrected_spectra': corrected_spectra,
        'recovered_components': H_matched,
        'recovered_weights': W_matched,
        'fitted_baselines': fitted_baselines,
        'permutation': col_ind,
    }
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_data, allow_pickle=True)
    
    # ============================================================
    # VISUALIZATION – 6 subplots
    # ============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # --- (a) Example mixed spectrum: observed vs corrected vs clean ---
    ax = axes[0, 0]
    idx = 0  # first mixture
    ax.plot(wavenumber, observed_spectra[idx], 'b-', alpha=0.6, label='Observed (with baseline+noise)')
    ax.plot(wavenumber, corrected_spectra[idx], 'r-', linewidth=1.5, label='Baseline corrected')
    ax.plot(wavenumber, clean_spectra[idx], 'k--', linewidth=1.5, label='Ground truth (clean)')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('(a) Baseline Correction Example')
    ax.legend(fontsize=8)
    
    # --- (b) Baseline correction detail ---
    ax = axes[0, 1]
    ax.plot(wavenumber, observed_spectra[idx], 'b-', alpha=0.5, label='Observed')
    ax.plot(wavenumber, fitted_baselines[idx], 'g-', linewidth=2, label='Fitted baseline')
    ax.plot(wavenumber, baselines[idx], 'm--', linewidth=2, label='True baseline')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('(b) True vs Fitted Baseline')
    ax.legend(fontsize=8)
    
    # --- (c) True vs recovered component spectra ---
    ax = axes[0, 2]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = ['Mineral A', 'Mineral B', 'Mineral C']
    comp_cc_list_local = []
    for i in range(n_components):
        cc = np.corrcoef(pure_components[i], H_matched[i])[0, 1]
        comp_cc_list_local.append(cc)
    for i in range(n_components):
        ax.plot(wavenumber, pure_components[i], '-', color=colors[i],
                linewidth=2, label=f'True {labels[i]}')
        ax.plot(wavenumber, H_matched[i], '--', color=colors[i],
                linewidth=1.5, alpha=0.8, label=f'Recovered (CC={comp_cc_list_local[i]:.3f})')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('(c) True vs Recovered Components')
    ax.legend(fontsize=7, ncol=2)
    
    # --- (d) Mixing proportions: true vs recovered (scatter) ---
    ax = axes[1, 0]
    for i in range(n_components):
        ax.scatter(true_weights[:, i], W_matched[:, i], color=colors[i],
                   s=40, label=f'{labels[i]}', alpha=0.8, edgecolors='k', linewidth=0.3)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Ideal')
    ax.set_xlabel('True Mixing Proportion')
    ax.set_ylabel('Recovered Mixing Proportion')
    ax.set_title('(d) Mixing Proportions: True vs Recovered')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # --- (e) Residual after baseline correction ---
    ax = axes[1, 1]
    residuals = corrected_spectra - clean_spectra
    for i in range(min(5, n_mixtures)):
        ax.plot(wavenumber, residuals[i], alpha=0.5, linewidth=0.8, label=f'Mix {i+1}')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Residual')
    ax.set_title(f'(e) Baseline Correction Residuals (PSNR={psnr_baseline:.1f} dB)')
    ax.legend(fontsize=7)
    
    # --- (f) Per-mixture weight recovery bar chart ---
    ax = axes[1, 2]
    mix_indices = np.arange(n_mixtures)
    bar_width = 0.12
    for i in range(n_components):
        ax.bar(mix_indices - bar_width + i * bar_width, true_weights[:, i],
               bar_width, color=colors[i], alpha=0.5, label=f'True {labels[i]}')
        ax.bar(mix_indices + i * bar_width, W_matched[:, i],
               bar_width, color=colors[i], edgecolor='k', linewidth=0.5,
               label=f'Rec {labels[i]}')
    ax.set_xlabel('Mixture Index')
    ax.set_ylabel('Weight')
    ax.set_title('(f) Weight Recovery per Mixture')
    # Simplify legend
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[:6], lbls[:6], fontsize=6, ncol=2)
    ax.set_xticks(mix_indices)
    
    plt.suptitle('Raman Spectral Unmixing & Baseline Correction (rampy + NMF)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir}/reconstruction_result.png")
    print(f"Metrics saved to {output_dir}/metrics.json")
    print(f"Ground truth saved to {output_dir}/ground_truth.npy")
    print(f"Reconstruction saved to {output_dir}/reconstruction.npy")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/rampy_unmix_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner files
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
    
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'unknown')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Outer function name: {func_name}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent function
    print("\n--- Running Agent Function ---")
    try:
        if args:
            agent_output = run_inversion(*args, **kwargs)
        else:
            # All arguments are in kwargs
            agent_output = run_inversion(**kwargs)
        print("Agent function completed successfully")
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check for chained execution (inner data)
    if inner_files:
        # Chained execution pattern
        print("\n--- Chained Execution Detected ---")
        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print(f"Inner function args: {len(inner_args)}, kwargs: {list(inner_kwargs.keys())}")
        
        # Execute the operator returned by run_inversion
        try:
            if inner_args:
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output(**inner_kwargs)
            print("Inner function completed successfully")
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        final_result = agent_output
        std_result = std_output
    
    # ============================================================
    # PREPARE DATA FOR EVALUATION
    # ============================================================
    
    # Extract ground truth data from kwargs (this is the input data structure)
    # The evaluate_results function expects a 'data' dict with ground truth
    
    # Get the input parameters
    if args:
        observed_spectra = args[0]
        wavenumber = args[1] if len(args) > 1 else kwargs.get('wavenumber')
        n_components = args[2] if len(args) > 2 else kwargs.get('n_components')
        pure_components = args[3] if len(args) > 3 else kwargs.get('pure_components')
    else:
        observed_spectra = kwargs.get('observed_spectra')
        wavenumber = kwargs.get('wavenumber')
        n_components = kwargs.get('n_components')
        pure_components = kwargs.get('pure_components')
    
    print(f"\nInput data shapes:")
    print(f"  observed_spectra: {observed_spectra.shape}")
    print(f"  wavenumber: {wavenumber.shape}")
    print(f"  n_components: {n_components}")
    print(f"  pure_components: {pure_components.shape}")
    
    # For evaluation, we need to construct or load the full ground truth data
    # The evaluate_results function requires:
    # - wavenumber, pure_components, true_weights, clean_spectra, baselines, observed_spectra, n_components
    
    # Since we don't have true_weights, clean_spectra, baselines in the pkl,
    # we'll do a simplified evaluation comparing agent vs standard output directly
    
    print("\n--- Evaluating Results ---")
    
    # Compare the recovered components and weights between agent and standard
    agent_result = final_result
    
    print("\nAgent output keys:", list(agent_result.keys()) if isinstance(agent_result, dict) else type(agent_result))
    print("Standard output keys:", list(std_result.keys()) if isinstance(std_result, dict) else type(std_result))
    
    # Calculate metrics by comparing agent output to standard output
    # Since both should produce similar results, we compare them directly
    
    # Component correlation
    agent_components = agent_result['recovered_components']
    std_components = std_result['recovered_components']
    
    agent_weights = agent_result['recovered_weights']
    std_weights = std_result['recovered_weights']
    
    agent_corrected = agent_result['corrected_spectra']
    std_corrected = std_result['corrected_spectra']
    
    print(f"\nComparing Agent vs Standard Results:")
    
    # Component correlation comparison
    comp_correlations = []
    for i in range(n_components):
        cc = np.corrcoef(agent_components[i], std_components[i])[0, 1]
        comp_correlations.append(cc)
        print(f"  Component {i+1} correlation (Agent vs Std): {cc:.6f}")
    
    mean_comp_cc = np.mean(comp_correlations)
    print(f"  Mean component correlation: {mean_comp_cc:.6f}")
    
    # Weights comparison
    weight_correlations = []
    for i in range(n_components):
        cc_w = np.corrcoef(agent_weights[:, i], std_weights[:, i])[0, 1]
        weight_correlations.append(cc_w)
        print(f"  Weight {i+1} correlation (Agent vs Std): {cc_w:.6f}")
    
    mean_weight_cc = np.mean(weight_correlations)
    print(f"  Mean weight correlation: {mean_weight_cc:.6f}")
    
    # Corrected spectra comparison
    mse_corrected = np.mean((agent_corrected - std_corrected) ** 2)
    max_val = np.max(np.abs(std_corrected))
    if mse_corrected > 0:
        psnr_corrected = 10 * np.log10(max_val ** 2 / mse_corrected)
    else:
        psnr_corrected = float('inf')
    print(f"  Corrected spectra PSNR (Agent vs Std): {psnr_corrected:.2f} dB")
    
    # Overall MSE for components
    mse_components = np.mean((agent_components - std_components) ** 2)
    print(f"  Components MSE: {mse_components:.8f}")
    
    # Overall MSE for weights
    mse_weights = np.mean((agent_weights - std_weights) ** 2)
    print(f"  Weights MSE: {mse_weights:.8f}")
    
    # ============================================================
    # VERIFICATION
    # ============================================================
    
    print("\n--- Verification ---")
    
    # Define thresholds for acceptable performance
    # Since this is comparing agent implementation to standard,
    # we expect very high correlation (>0.99) and low MSE
    
    COMPONENT_CC_THRESHOLD = 0.95  # Allow some tolerance
    WEIGHT_CC_THRESHOLD = 0.95
    PSNR_THRESHOLD = 20.0  # dB
    
    success = True
    
    if mean_comp_cc < COMPONENT_CC_THRESHOLD:
        print(f"WARNING: Mean component correlation {mean_comp_cc:.4f} < {COMPONENT_CC_THRESHOLD}")
        success = False
    else:
        print(f"PASS: Mean component correlation {mean_comp_cc:.4f} >= {COMPONENT_CC_THRESHOLD}")
    
    if mean_weight_cc < WEIGHT_CC_THRESHOLD:
        print(f"WARNING: Mean weight correlation {mean_weight_cc:.4f} < {WEIGHT_CC_THRESHOLD}")
        success = False
    else:
        print(f"PASS: Mean weight correlation {mean_weight_cc:.4f} >= {WEIGHT_CC_THRESHOLD}")
    
    if psnr_corrected < PSNR_THRESHOLD:
        print(f"WARNING: Corrected spectra PSNR {psnr_corrected:.2f} dB < {PSNR_THRESHOLD} dB")
        success = False
    else:
        print(f"PASS: Corrected spectra PSNR {psnr_corrected:.2f} dB >= {PSNR_THRESHOLD} dB")
    
    # Additional check: verify the permutation is valid
    agent_perm = agent_result['permutation']
    std_perm = std_result['permutation']
    print(f"\nAgent permutation: {agent_perm}")
    print(f"Standard permutation: {std_perm}")
    
    # Final summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Component Correlation: {mean_comp_cc:.4f}")
    print(f"Weight Correlation: {mean_weight_cc:.4f}")
    print(f"Spectra PSNR: {psnr_corrected:.2f} dB")
    
    if success:
        print("\nRESULT: PASSED - Agent performance is acceptable")
        sys.exit(0)
    else:
        print("\nRESULT: FAILED - Agent performance degraded")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)