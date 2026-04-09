import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the referee evaluation function verbatim
# ============================================================

def gaussian_peak(energy, center, amplitude, sigma):
    """Gaussian peak profile for detector-broadened XRF line."""
    return amplitude * np.exp(-0.5 * ((energy - center) / sigma) ** 2)

def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def generate_element_spectrum(energy, element, concentration, det_sigma, xrf_lines):
    """
    Generate XRF spectrum for a single element.
    Each characteristic line is a Gaussian broadened by detector resolution.
    """
    spectrum = np.zeros_like(energy)
    if element not in xrf_lines:
        return spectrum
    
    for line_name, line_energy, rel_intensity in xrf_lines[element]:
        amplitude = concentration * rel_intensity
        spectrum += gaussian_peak(energy, line_energy, amplitude, det_sigma)
    
    return spectrum

def evaluate_results(gt_concentrations, recon_result, clean_spectrum, metadata, 
                     results_dir, e_min, e_max, detector_fwhm):
    """
    Compute evaluation metrics, save results, and generate visualizations.
    """
    refined = recon_result['refined_concentrations']
    recon_spectrum = recon_result['recon_spectrum']
    elements = list(gt_concentrations.keys())
    energy = metadata['energy']
    noisy_spectrum = metadata.get('noisy_spectrum', None)
    xrf_lines = metadata['xrf_lines']
    
    if noisy_spectrum is None:
        noisy_spectrum = clean_spectrum  # fallback
    
    metrics = {}
    
    # Per-element concentration errors
    gt_array = []
    fitted_array = []
    rel_errors = []
    
    for el in elements:
        gt_val = gt_concentrations[el]
        fit_val = refined.get(el, 0.0)
        gt_array.append(gt_val)
        fitted_array.append(fit_val)
        re = abs(fit_val - gt_val) / gt_val * 100.0 if gt_val > 0 else 0.0
        rel_errors.append(re)
        metrics[f'{el}_gt'] = gt_val
        metrics[f'{el}_fitted'] = round(fit_val, 4)
        metrics[f'{el}_RE_pct'] = round(re, 4)
    
    gt_array = np.array(gt_array)
    fitted_array = np.array(fitted_array)
    
    # Overall concentration metrics
    metrics['mean_RE_pct'] = float(np.mean(rel_errors))
    metrics['max_RE_pct'] = float(np.max(rel_errors))
    
    # Concentration CC
    if len(gt_array) > 1:
        cc = float(np.corrcoef(gt_array, fitted_array)[0, 1])
    else:
        cc = 1.0
    metrics['concentration_CC'] = cc
    
    # Concentration R²
    ss_res = np.sum((gt_array - fitted_array) ** 2)
    ss_tot = np.sum((gt_array - np.mean(gt_array)) ** 2)
    metrics['concentration_R2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    # Concentration PSNR
    data_range = gt_array.max() - gt_array.min() if len(gt_array) > 1 else gt_array.max()
    mse_conc = np.mean((gt_array - fitted_array) ** 2)
    if mse_conc > 0 and data_range > 0:
        metrics['concentration_PSNR'] = float(10 * np.log10(data_range ** 2 / mse_conc))
    else:
        metrics['concentration_PSNR'] = float('inf')
    
    # Spectrum-level metrics
    data_range_spec = clean_spectrum.max() - clean_spectrum.min()
    mse_spec = np.mean((clean_spectrum - recon_spectrum) ** 2)
    if mse_spec > 0 and data_range_spec > 0:
        metrics['spectrum_PSNR'] = float(10 * np.log10(data_range_spec ** 2 / mse_spec))
    else:
        metrics['spectrum_PSNR'] = float('inf')
    
    cc_spec = float(np.corrcoef(clean_spectrum.flatten(), recon_spectrum.flatten())[0, 1])
    metrics['spectrum_CC'] = cc_spec
    
    rmse_spec = float(np.sqrt(mse_spec))
    metrics['spectrum_RMSE'] = rmse_spec
    
    # Print metrics
    print(f"\n[EVAL] === Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Generate visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    _visualize_results(energy, noisy_spectrum, clean_spectrum, recon_result, 
                       gt_concentrations, metrics, vis_path, e_min, e_max, 
                       detector_fwhm, xrf_lines)
    
    # Save arrays
    gt_conc_array = np.array([gt_concentrations[el] for el in elements])
    fit_conc_array = np.array([refined[el] for el in elements])
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_conc_array)
    np.save(os.path.join(results_dir, "reconstruction.npy"), fit_conc_array)
    
    return metrics

def _visualize_results(energy, noisy_spectrum, clean_spectrum, recon_result, 
                       gt_concentrations, metrics, save_path, e_min, e_max,
                       detector_fwhm, xrf_lines):
    """Generate 4-panel visualization for XRF deconvolution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    recon_spectrum = recon_result['recon_spectrum']
    refined = recon_result['refined_concentrations']
    basis_spectra = recon_result['basis_spectra']
    fitted_bg = recon_result['fitted_background']
    elements = list(gt_concentrations.keys())
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))
    
    # Panel 1
    ax1 = axes[0, 0]
    ax1.plot(energy, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.5, label='Noisy data')
    ax1.plot(energy, clean_spectrum, 'b-', alpha=0.8, linewidth=1.0, label='Clean total')
    det_sigma = fwhm_to_sigma(detector_fwhm)
    for i, el in enumerate(elements):
        el_spec = generate_element_spectrum(energy, el, gt_concentrations[el], det_sigma, xrf_lines)
        ax1.fill_between(energy, 0, el_spec, alpha=0.3, color=colors[i], label=f'{el} (GT)')
    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Intensity (counts)')
    ax1.set_title('(a) GT XRF Spectrum — Element Contributions')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_xlim(e_min, e_max)
    
    # Panel 2
    ax2 = axes[0, 1]
    ax2.plot(energy, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.5, label='Noisy data')
    ax2.plot(energy, recon_spectrum, 'r-', alpha=0.8, linewidth=1.0, label='Fitted total')
    ax2.plot(energy, fitted_bg, 'g--', alpha=0.6, linewidth=1.0, label='Background')
    for i, el in enumerate(elements):
        el_spec = refined[el] * basis_spectra[el]
        ax2.fill_between(energy, 0, el_spec, alpha=0.3, color=colors[i], label=f'{el} (fit)')
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel('Intensity (counts)')
    psnr = metrics.get('spectrum_PSNR', 0)
    cc = metrics.get('spectrum_CC', 0)
    ax2.set_title(f'(b) Fitted XRF Decomposition — PSNR={psnr:.2f} dB, CC={cc:.4f}')
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_xlim(e_min, e_max)
    
    # Panel 3
    ax3 = axes[1, 0]
    residual = noisy_spectrum - recon_spectrum
    ax3.plot(energy, residual, 'k-', alpha=0.5, linewidth=0.5)
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.fill_between(energy, -2*np.std(residual), 2*np.std(residual), alpha=0.1, color='blue')
    ax3.set_xlabel('Energy (keV)')
    ax3.set_ylabel('Residual')
    ax3.set_title(f'(c) Fit Residual — RMSE={metrics.get("spectrum_RMSE", 0):.4f}')
    ax3.set_xlim(e_min, e_max)
    
    # Panel 4
    ax4 = axes[1, 1]
    x = np.arange(len(elements))
    width = 0.35
    gt_vals = [gt_concentrations[el] for el in elements]
    fit_vals = [refined.get(el, 0) for el in elements]
    
    ax4.bar(x - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, fit_vals, width, label='Fitted', color='coral', alpha=0.8)
    
    for i, el in enumerate(elements):
        re = metrics.get(f'{el}_RE_pct', 0)
        ax4.annotate(f'{re:.1f}%', (x[i] + width/2, fit_vals[i] + 2), ha='center', fontsize=7)
    
    ax4.set_xlabel('Element')
    ax4.set_ylabel('Concentration (a.u.)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(elements)
    mean_re = metrics.get('mean_RE_pct', 0)
    conc_cc = metrics.get('concentration_CC', 0)
    ax4.set_title(f'(d) Concentration Recovery — CC={conc_cc:.4f}, Mean RE={mean_re:.2f}%')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = ['/data/yjh/pyxrf_fluor_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"[INFO] Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    args = outer_data['args']
    kwargs = outer_data['kwargs']
    std_output = outer_data['output']
    
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(args)}, Number of kwargs: {len(kwargs)}")
    
    # Determine execution pattern
    is_chained = len(inner_paths) > 0
    
    if is_chained:
        # Pattern 2: Chained Execution
        print("[INFO] Chained execution detected.")
        agent_output = run_inversion(*args, **kwargs)
        
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data['args']
            inner_kwargs = inner_data['kwargs']
            std_result = inner_data['output']
            
            final_result = agent_output(*inner_args, **inner_kwargs)
    else:
        # Pattern 1: Direct Execution
        print("[INFO] Direct execution pattern.")
        try:
            agent_output = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] run_inversion failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        final_result = agent_output
        std_result = std_output
    
    print("[INFO] Agent execution completed successfully.")
    
    # ============================================================
    # Extract ground truth info from metadata for evaluation
    # ============================================================
    # The inputs to run_inversion are (noisy_spectrum, metadata)
    noisy_spectrum = args[0]
    metadata = args[1]
    
    # We need gt_concentrations and clean_spectrum for evaluate_results.
    # These might be stored in metadata or we need to derive them from std_result.
    # Check metadata for these fields
    gt_concentrations = metadata.get('gt_concentrations', None)
    clean_spectrum = metadata.get('clean_spectrum', None)
    e_min = metadata.get('e_min', None)
    e_max = metadata.get('e_max', None)
    detector_fwhm = metadata.get('det_fwhm', 0.15)
    
    # If gt_concentrations not in metadata, try to extract from the energy range
    if gt_concentrations is None:
        # Try to find it in metadata under different keys
        for key in ['concentrations', 'true_concentrations', 'ground_truth']:
            if key in metadata:
                gt_concentrations = metadata[key]
                break
    
    if gt_concentrations is None:
        # Fall back: use std_result refined_concentrations as "ground truth proxy"
        # This is not ideal but allows us to compare agent vs standard
        print("[WARN] gt_concentrations not found in metadata. Using std refined_concentrations as reference.")
        gt_concentrations = std_result.get('refined_concentrations', std_result.get('nnls_concentrations', {}))
    
    if clean_spectrum is None:
        # Try to find clean_spectrum in metadata
        for key in ['clean_spectrum', 'true_spectrum', 'noiseless_spectrum']:
            if key in metadata:
                clean_spectrum = metadata[key]
                break
    
    if clean_spectrum is None:
        # Build clean spectrum from gt_concentrations if we have them
        print("[WARN] clean_spectrum not found. Attempting to reconstruct from GT concentrations.")
        try:
            energy = metadata['energy']
            elements = metadata['elements']
            xrf_lines = metadata['xrf_lines']
            det_sigma = fwhm_to_sigma(detector_fwhm)
            clean_spectrum = np.zeros_like(energy)
            for el in elements:
                if el in gt_concentrations:
                    clean_spectrum += generate_element_spectrum(energy, el, gt_concentrations[el], det_sigma, xrf_lines)
        except Exception as e:
            print(f"[WARN] Could not reconstruct clean spectrum: {e}")
            # Use the noisy spectrum as fallback
            clean_spectrum = noisy_spectrum.copy()
    
    if e_min is None:
        energy = metadata['energy']
        e_min = float(energy.min())
    if e_max is None:
        energy = metadata['energy']
        e_max = float(energy.max())
    
    # Add noisy_spectrum to metadata for evaluation visualization
    metadata_eval = dict(metadata)
    metadata_eval['noisy_spectrum'] = noisy_spectrum
    
    # ============================================================
    # Evaluate agent result
    # ============================================================
    print("\n" + "="*60)
    print("[EVAL] Evaluating AGENT result...")
    print("="*60)
    
    results_dir_agent = './eval_results_agent'
    try:
        metrics_agent = evaluate_results(
            gt_concentrations, final_result, clean_spectrum, metadata_eval,
            results_dir_agent, e_min, e_max, detector_fwhm
        )
    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================
    # Evaluate standard result
    # ============================================================
    print("\n" + "="*60)
    print("[EVAL] Evaluating STANDARD result...")
    print("="*60)
    
    results_dir_std = './eval_results_std'
    try:
        metrics_std = evaluate_results(
            gt_concentrations, std_result, clean_spectrum, metadata_eval,
            results_dir_std, e_min, e_max, detector_fwhm
        )
    except Exception as e:
        print(f"[ERROR] Standard evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================
    # Compare results
    # ============================================================
    print("\n" + "="*60)
    print("[COMPARISON] Agent vs Standard")
    print("="*60)
    
    # Key metrics to compare (higher is better for these)
    higher_is_better_keys = ['concentration_CC', 'concentration_R2', 'concentration_PSNR', 
                              'spectrum_PSNR', 'spectrum_CC']
    # Lower is better for these
    lower_is_better_keys = ['mean_RE_pct', 'max_RE_pct', 'spectrum_RMSE']
    
    all_pass = True
    margin = 0.10  # 10% margin
    
    for key in higher_is_better_keys:
        agent_val = metrics_agent.get(key, None)
        std_val = metrics_std.get(key, None)
        if agent_val is not None and std_val is not None:
            # Handle inf cases
            if np.isinf(agent_val) and np.isinf(std_val):
                status = "PASS (both inf)"
            elif np.isinf(std_val):
                status = "FAIL (std is inf, agent is not)"
                all_pass = False
            elif np.isinf(agent_val):
                status = "PASS (agent is inf)"
            else:
                threshold = std_val * (1.0 - margin) if std_val > 0 else std_val * (1.0 + margin)
                if agent_val >= threshold:
                    status = "PASS"
                else:
                    status = "FAIL"
                    all_pass = False
            print(f"  {key}: Agent={agent_val:.6f}, Std={std_val:.6f} -> {status}")
    
    for key in lower_is_better_keys:
        agent_val = metrics_agent.get(key, None)
        std_val = metrics_std.get(key, None)
        if agent_val is not None and std_val is not None:
            threshold = std_val * (1.0 + margin) if std_val > 0 else std_val * (1.0 - margin)
            if agent_val <= threshold:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
            print(f"  {key}: Agent={agent_val:.6f}, Std={std_val:.6f} -> {status}")
    
    print("\n" + "="*60)
    if all_pass:
        print("[RESULT] ALL CHECKS PASSED. Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("[RESULT] SOME CHECKS FAILED. Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)