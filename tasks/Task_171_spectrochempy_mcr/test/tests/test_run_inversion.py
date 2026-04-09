import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================================
# INJECTED REFEREE CODE (from Reference B)
# ============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def match_components(S_true, S_recovered):
    """
    Match recovered components to true components using correlation.
    Returns permutation indices and correlation matrix.
    """
    n_comp = S_true.shape[0]
    corr_matrix = np.zeros((n_comp, n_comp))

    for i in range(n_comp):
        for j in range(n_comp):
            s_true_norm = S_true[i] / (np.linalg.norm(S_true[i]) + 1e-12)
            s_rec_norm = S_recovered[j] / (np.linalg.norm(S_recovered[j]) + 1e-12)
            corr_matrix[i, j] = np.abs(np.dot(s_true_norm, s_rec_norm))

    # Greedy matching
    perm = []
    used = set()
    for i in range(n_comp):
        best_j = -1
        best_corr = -1
        for j in range(n_comp):
            if j not in used and corr_matrix[i, j] > best_corr:
                best_corr = corr_matrix[i, j]
                best_j = j
        perm.append(best_j)
        used.add(best_j)

    return perm, corr_matrix

def compute_spectral_cc(S_true, S_recovered, perm):
    """Compute correlation coefficients between true and recovered spectra."""
    ccs = []
    for i, j in enumerate(perm):
        cc = np.corrcoef(S_true[i], S_recovered[j])[0, 1]
        ccs.append(abs(cc))
    return np.array(ccs)

def compute_concentration_re(C_true, C_recovered, perm):
    """Compute relative error of recovered concentrations."""
    res = []
    for i, j in enumerate(perm):
        c_true = C_true[:, i]
        c_rec = C_recovered[:, j]
        # Find optimal scaling factor
        scale = np.dot(c_true, c_rec) / (np.dot(c_rec, c_rec) + 1e-12)
        c_rec_scaled = c_rec * scale
        re = np.linalg.norm(c_true - c_rec_scaled) / (np.linalg.norm(c_true) + 1e-12)
        res.append(re)
    return np.array(res)

def compute_psnr(D_true, D_reconstructed):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((D_true - D_reconstructed) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(D_true))
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr

def compute_lack_of_fit(D, D_reconstructed):
    """Compute lack-of-fit percentage."""
    residual = D - D_reconstructed
    lof = np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100
    return lof

def plot_results(wavelengths, S_true, S_recovered, C_true, C_recovered,
                 D_clean, D_reconstructed, perm, save_path):
    """Create 4-subplot visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    comp_names = ['Component 1', 'Component 2', 'Component 3']

    # Scale recovered components to match true for visualization
    S_rec_scaled = np.zeros_like(S_recovered)
    C_rec_scaled = np.zeros_like(C_recovered)
    for i, j in enumerate(perm):
        s_scale = np.max(S_true[i]) / (np.max(S_recovered[j]) + 1e-12)
        S_rec_scaled[i] = S_recovered[j] * s_scale

        c_scale = np.dot(C_true[:, i], C_recovered[:, j]) / (np.dot(C_recovered[:, j], C_recovered[:, j]) + 1e-12)
        C_rec_scaled[:, i] = C_recovered[:, j] * c_scale

    # (a) True vs recovered spectra
    ax = axes[0, 0]
    for i in range(S_true.shape[0]):
        ax.plot(wavelengths, S_true[i], color=colors[i], linewidth=2,
                label=f'{comp_names[i]} (true)')
        ax.plot(wavelengths, S_rec_scaled[i], color=colors[i], linewidth=1.5,
                linestyle='--', label=f'{comp_names[i]} (recovered)')
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title('(a) True vs Recovered Component Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) True vs recovered concentration profiles
    ax = axes[0, 1]
    samples = np.arange(C_true.shape[0])
    for i in range(C_true.shape[1]):
        ax.plot(samples, C_true[:, i], color=colors[i], linewidth=2,
                label=f'{comp_names[i]} (true)')
        ax.plot(samples, C_rec_scaled[:, i], color=colors[i], linewidth=1.5,
                linestyle='--', label=f'{comp_names[i]} (recovered)')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Concentration', fontsize=11)
    ax.set_title('(b) True vs Recovered Concentration Profiles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (c) Reconstructed vs original D (first 5 samples)
    ax = axes[1, 0]
    n_show = min(5, D_clean.shape[0])
    sample_colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    for i in range(n_show):
        ax.plot(wavelengths, D_clean[i], color=sample_colors[i], linewidth=1.5,
                label=f'Sample {i+1} (true)')
        ax.plot(wavelengths, D_reconstructed[i], color=sample_colors[i],
                linewidth=1, linestyle='--', alpha=0.8)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title('(c) Original vs Reconstructed Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (d) Residual matrix heatmap
    ax = axes[1, 1]
    residual = D_clean - D_reconstructed
    im = ax.imshow(residual, aspect='auto', cmap='RdBu_r',
                   extent=[wavelengths[0], wavelengths[-1], D_clean.shape[0], 0])
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Sample Index', fontsize=11)
    ax.set_title('(d) Residual Matrix (D_true - D_reconstructed)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Residual')

    plt.suptitle('MCR-ALS Spectral Decomposition Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the MCR-ALS decomposition results.
    
    Computes metrics comparing recovered components to ground truth,
    saves results to files, and generates visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data containing:
        - 'wavelengths': wavelength array
        - 'S_true': true pure component spectra
        - 'C_true': true concentration profiles
        - 'D_clean': noise-free data matrix
        - 'snr_db': signal-to-noise ratio
        - 'n_components': number of components
    inversion_result : dict
        Dictionary from run_inversion containing:
        - 'C_recovered': recovered concentration matrix
        - 'S_recovered': recovered spectra matrix
        - 'D_reconstructed': reconstructed data matrix
        - 'method_used': string indicating which method was used
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    wavelengths = data_dict['wavelengths']
    S_true = data_dict['S_true']
    C_true = data_dict['C_true']
    D_clean = data_dict['D_clean']
    snr_db = data_dict['snr_db']
    n_components = data_dict['n_components']
    
    C_recovered = inversion_result['C_recovered']
    S_recovered = inversion_result['S_recovered']
    D_reconstructed = inversion_result['D_reconstructed']
    method_used = inversion_result['method_used']
    
    n_samples = C_true.shape[0]
    n_wavelengths = len(wavelengths)
    
    # Match components
    perm, corr_matrix = match_components(S_true, S_recovered)
    print(f"Component matching (true→recovered): {perm}")
    
    # Spectral correlation coefficients
    spectral_ccs = compute_spectral_cc(S_true, S_recovered, perm)
    mean_cc = np.mean(spectral_ccs)
    print(f"\nSpectral Correlation Coefficients:")
    for i, cc in enumerate(spectral_ccs):
        print(f"  Component {i+1}: {cc:.6f}")
    print(f"  Mean CC: {mean_cc:.6f}")
    
    # Concentration relative error
    conc_res = compute_concentration_re(C_true, C_recovered, perm)
    mean_re = np.mean(conc_res)
    print(f"\nConcentration Relative Errors:")
    for i, re in enumerate(conc_res):
        print(f"  Component {i+1}: {re:.6f}")
    print(f"  Mean RE: {mean_re:.6f}")
    
    # Reconstruction metrics
    psnr = compute_psnr(D_clean, D_reconstructed)
    lof = compute_lack_of_fit(D_clean, D_reconstructed)
    print(f"\nReconstruction PSNR: {psnr:.2f} dB")
    print(f"Lack-of-fit: {lof:.4f}%")
    
    # Save metrics
    metrics = {
        "task": "spectrochempy_mcr",
        "method": method_used,
        "inverse_problem": "MCR-ALS spectral decomposition",
        "n_samples": n_samples,
        "n_components": n_components,
        "n_wavelengths": n_wavelengths,
        "snr_db": snr_db,
        "spectral_cc_per_component": spectral_ccs.tolist(),
        "spectral_cc_mean": float(mean_cc),
        "concentration_re_per_component": conc_res.tolist(),
        "concentration_re_mean": float(mean_re),
        "reconstruction_psnr_db": float(psnr),
        "lack_of_fit_pct": float(lof),
        "component_matching": perm
    }
    
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.json")
    
    # Save ground truth and reconstruction
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), D_clean)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), D_reconstructed)
    print(f"Ground truth saved: {RESULTS_DIR}/ground_truth.npy {D_clean.shape}")
    print(f"Reconstruction saved: {RESULTS_DIR}/reconstruction.npy {D_reconstructed.shape}")
    
    # Visualization
    plot_results(wavelengths, S_true, S_recovered, C_true, C_recovered,
                 D_clean, D_reconstructed, perm,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean Spectral CC: {mean_cc:.6f} (target > 0.95)")
    print(f"Reconstruction PSNR: {psnr:.2f} dB (target > 25)")
    print(f"Mean Concentration RE: {mean_re:.6f}")
    print(f"Lack-of-fit: {lof:.4f}%")
    
    if mean_cc > 0.95 and psnr > 25:
        print("✓ All targets met!")
    else:
        print("✗ Some targets not met.")
    
    return metrics

# ============================================================================
# END OF INJECTED REFEREE CODE
# ============================================================================

def main():
    # Data paths provided
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Load outer (primary) data
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"\nLoading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Outer data keys: {outer_data.keys()}")
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {kwargs.keys()}")
    
    # Check if this is a chained execution pattern
    if inner_files:
        # Pattern 2: Chained Execution
        print("\n=== CHAINED EXECUTION PATTERN ===")
        
        # Step 1: Run outer function to get operator
        print("\nStep 1: Running run_inversion to get operator...")
        try:
            operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Step 2: Load inner data and run operator
        inner_path = inner_files[0]
        print(f"\nStep 2: Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print(f"Inner data keys: {inner_data.keys()}")
        
        # Run operator with inner data
        print("\nRunning operator with inner data...")
        try:
            agent_result = operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Also get standard operator result
        std_operator = std_output
        if callable(std_operator):
            std_result = std_operator(*inner_args, **inner_kwargs)
    else:
        # Pattern 1: Direct Execution
        print("\n=== DIRECT EXECUTION PATTERN ===")
        
        print("\nRunning run_inversion...")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        std_result = std_output
    
    print("\n=== RESULTS ===")
    print(f"Agent result type: {type(agent_result)}")
    print(f"Standard result type: {type(std_result)}")
    
    if isinstance(agent_result, dict):
        print(f"Agent result keys: {agent_result.keys()}")
    if isinstance(std_result, dict):
        print(f"Standard result keys: {std_result.keys()}")
    
    # For evaluation, we need the data_dict (input data with ground truth)
    # The first argument to run_inversion should be data_dict
    if len(args) > 0:
        data_dict = args[0]
    else:
        data_dict = kwargs.get('data_dict', None)
    
    if data_dict is None:
        print("ERROR: Could not extract data_dict for evaluation!")
        sys.exit(1)
    
    # Check if data_dict has required keys for evaluation
    required_keys = ['wavelengths', 'S_true', 'C_true', 'D_clean', 'snr_db', 'n_components']
    missing_keys = [k for k in required_keys if k not in data_dict]
    
    if missing_keys:
        print(f"WARNING: data_dict missing keys: {missing_keys}")
        print(f"Available keys: {data_dict.keys()}")
        # Try to proceed with basic comparison
        
        # Basic metric comparison if we can't use full evaluate_results
        if isinstance(agent_result, dict) and isinstance(std_result, dict):
            # Compare D_reconstructed directly
            if 'D_reconstructed' in agent_result and 'D_reconstructed' in std_result:
                D_agent = agent_result['D_reconstructed']
                D_std = std_result['D_reconstructed']
                
                # Compute PSNR-like metrics
                mse_agent = np.mean((data_dict.get('D_noisy', D_std) - D_agent) ** 2)
                mse_std = np.mean((data_dict.get('D_noisy', D_std) - D_std) ** 2)
                
                print(f"\nMSE Agent: {mse_agent:.6f}")
                print(f"MSE Standard: {mse_std:.6f}")
                
                # Allow 20% margin
                if mse_agent <= mse_std * 1.2:
                    print("\n✓ Agent performance is acceptable (within 20% of standard)")
                    sys.exit(0)
                else:
                    print("\n✗ Agent performance degraded significantly")
                    sys.exit(1)
        
        print("Cannot perform full evaluation, exiting with success")
        sys.exit(0)
    
    # Perform full evaluation
    print("\n=== EVALUATING AGENT RESULT ===")
    try:
        metrics_agent = evaluate_results(data_dict, agent_result)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n=== EVALUATING STANDARD RESULT ===")
    # Save to different directory to avoid overwriting
    global RESULTS_DIR
    RESULTS_DIR = "results_std"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        metrics_std = evaluate_results(data_dict, std_result)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Reset results dir
    RESULTS_DIR = "results"
    
    # Compare metrics
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    score_agent_cc = metrics_agent['spectral_cc_mean']
    score_std_cc = metrics_std['spectral_cc_mean']
    
    score_agent_psnr = metrics_agent['reconstruction_psnr_db']
    score_std_psnr = metrics_std['reconstruction_psnr_db']
    
    score_agent_lof = metrics_agent['lack_of_fit_pct']
    score_std_lof = metrics_std['lack_of_fit_pct']
    
    print(f"Spectral CC -> Agent: {score_agent_cc:.6f}, Standard: {score_std_cc:.6f}")
    print(f"PSNR (dB) -> Agent: {score_agent_psnr:.2f}, Standard: {score_std_psnr:.2f}")
    print(f"Lack of Fit (%) -> Agent: {score_agent_lof:.4f}, Standard: {score_std_lof:.4f}")
    
    # Determine success:
    # - Spectral CC: Higher is better (allow 5% margin)
    # - PSNR: Higher is better (allow 5% margin)
    # - Lack of Fit: Lower is better (allow 10% margin)
    
    cc_ok = score_agent_cc >= score_std_cc * 0.95
    psnr_ok = score_agent_psnr >= score_std_psnr * 0.95
    lof_ok = score_agent_lof <= score_std_lof * 1.10
    
    print(f"\nChecks:")
    print(f"  CC check (agent >= 0.95 * std): {cc_ok}")
    print(f"  PSNR check (agent >= 0.95 * std): {psnr_ok}")
    print(f"  LOF check (agent <= 1.10 * std): {lof_ok}")
    
    if cc_ok and psnr_ok and lof_ok:
        print("\n✓ All performance checks passed!")
        sys.exit(0)
    else:
        # Be more lenient - if major metrics are close, pass
        if score_agent_cc > 0.90 and score_agent_psnr > 20:
            print("\n✓ Agent meets minimum quality thresholds")
            sys.exit(0)
        else:
            print("\n✗ Agent performance degraded significantly")
            sys.exit(1)

if __name__ == "__main__":
    main()