import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Import the target function
from agent_run_inversion import run_inversion

# Inject the evaluate_results function (Reference B - verbatim)
def evaluate_results(gt_freq_array, recon_result, clean_spectrum, metadata, results_dir):
    """
    Compute evaluation metrics and generate visualizations for peakbagging.
    """
    matched_gt = recon_result['matched_gt']
    matched_fitted = recon_result['matched_fitted']
    recon_spectrum = recon_result['recon_spectrum']
    fitted_freqs = recon_result['fitted_freqs']
    freqs = metadata['freqs']
    noisy_spectrum = metadata.get('noisy_spectrum', None)
    delta_nu = metadata['delta_nu']
    freq_min = metadata['freq_min']
    freq_max = metadata['freq_max']
    gt_modes = metadata['gt_modes']
    smoothed = metadata['smoothed_spectrum']
    
    metrics = {}
    
    if len(matched_gt) > 0:
        freq_errors = np.abs(matched_fitted - matched_gt)
        rel_errors = freq_errors / matched_gt * 100
        
        metrics['mean_freq_error_uHz'] = float(np.mean(freq_errors))
        metrics['median_freq_error_uHz'] = float(np.median(freq_errors))
        metrics['max_freq_error_uHz'] = float(np.max(freq_errors))
        metrics['mean_relative_error_pct'] = float(np.mean(rel_errors))
        metrics['median_relative_error_pct'] = float(np.median(rel_errors))
        
        if len(matched_gt) > 1:
            cc = float(np.corrcoef(matched_gt, matched_fitted)[0, 1])
        else:
            cc = 1.0
        metrics['frequency_CC'] = cc
        
        ss_res = np.sum((matched_gt - matched_fitted) ** 2)
        ss_tot = np.sum((matched_gt - np.mean(matched_gt)) ** 2)
        metrics['frequency_R2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        metrics['n_gt_modes'] = len(gt_freq_array)
        metrics['n_detected'] = len(fitted_freqs)
        metrics['n_matched'] = len(matched_gt)
        metrics['detection_rate'] = float(len(matched_gt) / len(gt_freq_array))
    
    if clean_spectrum is not None and recon_spectrum is not None:
        data_range = clean_spectrum.max() - clean_spectrum.min()
        mse = np.mean((clean_spectrum - recon_spectrum) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(data_range ** 2 / mse)
        else:
            psnr = float('inf')
        metrics['spectrum_PSNR'] = float(psnr)
        
        cc_spec = float(np.corrcoef(clean_spectrum, recon_spectrum)[0, 1])
        metrics['spectrum_CC'] = cc_spec
    
    if len(matched_gt) >= 4:
        sorted_fitted = np.sort(matched_fitted)
        diffs = np.diff(sorted_fitted)
        dnu_candidates = diffs[(diffs > delta_nu * 0.7) & (diffs < delta_nu * 1.3)]
        if len(dnu_candidates) > 0:
            fitted_dnu = np.median(dnu_candidates)
            metrics['fitted_delta_nu'] = float(fitted_dnu)
            metrics['delta_nu_error_pct'] = float(abs(fitted_dnu - delta_nu) / delta_nu * 100)
    
    print(f"\n[EVAL] === Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics -> {metrics_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax1.plot(freqs, clean_spectrum, 'b-', alpha=0.8, linewidth=0.5, label='Clean spectrum')
    for (n, l), freq in sorted(gt_modes.items()):
        colors = {0: 'red', 1: 'green', 2: 'blue'}
        ax1.axvline(freq, color=colors.get(l, 'gray'), alpha=0.4, linewidth=0.5)
    ax1.set_xlabel('Frequency (μHz)')
    ax1.set_ylabel('Power (ppm²/μHz)')
    ax1.set_title('(a) Ground Truth Power Spectrum + Mode Frequencies')
    ax1.set_xlim(freq_min, freq_max)
    ax1.legend(fontsize=8)
    
    ax2 = axes[0, 1]
    if noisy_spectrum is not None:
        ax2.plot(freqs, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.3, label='Noisy')
    ax2.plot(freqs, smoothed, 'b-', alpha=0.8, linewidth=0.8, label='Smoothed')
    ax2.plot(freqs, recon_result['bg_estimate'], 'r--', alpha=0.7, linewidth=1, label='Background')
    for ff in fitted_freqs:
        ax2.axvline(ff, color='orange', alpha=0.4, linewidth=0.5)
    ax2.set_xlabel('Frequency (μHz)')
    ax2.set_ylabel('Power (ppm²/μHz)')
    ax2.set_title(f'(b) Observed Spectrum + {len(fitted_freqs)} Detected Peaks')
    ax2.set_xlim(freq_min, freq_max)
    ax2.legend(fontsize=8)
    
    ax3 = axes[1, 0]
    ax3.plot(freqs, clean_spectrum, 'b-', alpha=0.6, linewidth=0.8, label='Clean GT')
    ax3.plot(freqs, recon_spectrum, 'r-', alpha=0.6, linewidth=0.8, label='Reconstructed')
    ax3.set_xlabel('Frequency (μHz)')
    ax3.set_ylabel('Power (ppm²/μHz)')
    psnr_val = metrics.get('spectrum_PSNR', 0)
    cc_val = metrics.get('spectrum_CC', 0)
    ax3.set_title(f'(c) Spectrum Reconstruction — PSNR={psnr_val:.2f} dB, CC={cc_val:.4f}')
    ax3.set_xlim(freq_min, freq_max)
    ax3.legend(fontsize=8)
    
    ax4 = axes[1, 1]
    if len(matched_gt) > 0:
        ax4.scatter(matched_gt, matched_fitted, c='blue', s=50, zorder=5, label='Matched modes')
        fmin_plot, fmax_plot = matched_gt.min() - 20, matched_gt.max() + 20
        ax4.plot([fmin_plot, fmax_plot], [fmin_plot, fmax_plot], 'k--', alpha=0.5, label='Perfect match')
        
        errors = matched_fitted - matched_gt
        for i in range(len(matched_gt)):
            ax4.annotate(f'{errors[i]:.2f}', (matched_gt[i], matched_fitted[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)
        
        freq_cc = metrics.get('frequency_CC', 0)
        mean_re = metrics.get('mean_relative_error_pct', 0)
        det_rate = metrics.get('detection_rate', 0)
        ax4.set_title(f'(d) Freq Match — CC={freq_cc:.6f}, RE={mean_re:.4f}%, Det={det_rate:.1%}')
    ax4.set_xlabel('Ground Truth Frequency (μHz)')
    ax4.set_ylabel('Fitted Frequency (μHz)')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization -> {vis_path}")
    
    np.save(os.path.join(results_dir, "reconstruction.npy"), matched_fitted)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_freq_array)
    if noisy_spectrum is not None:
        np.save(os.path.join(results_dir, "noisy_spectrum.npy"), noisy_spectrum)
    np.save(os.path.join(results_dir, "clean_spectrum.npy"), clean_spectrum)
    np.save(os.path.join(results_dir, "recon_spectrum.npy"), recon_spectrum)
    
    return metrics


def compute_composite_score(metrics):
    """
    Compute a composite score from the metrics dict.
    Higher is better.
    """
    score = 0.0
    components = 0
    
    # frequency_CC: higher is better, weight heavily
    if 'frequency_CC' in metrics:
        score += metrics['frequency_CC'] * 40.0
        components += 1
    
    # detection_rate: higher is better
    if 'detection_rate' in metrics:
        score += metrics['detection_rate'] * 30.0
        components += 1
    
    # spectrum_CC: higher is better
    if 'spectrum_CC' in metrics:
        score += metrics['spectrum_CC'] * 20.0
        components += 1
    
    # spectrum_PSNR: higher is better (normalize roughly)
    if 'spectrum_PSNR' in metrics:
        score += min(metrics['spectrum_PSNR'] / 50.0, 1.0) * 10.0
        components += 1
    
    # mean_relative_error_pct: lower is better (penalize)
    if 'mean_relative_error_pct' in metrics:
        penalty = min(metrics['mean_relative_error_pct'], 10.0) / 10.0
        score -= penalty * 5.0
    
    return score


def main():
    data_paths = ['/data/yjh/pbjam_astero_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Determine execution pattern
    std_data_dir = os.path.dirname(data_paths[0])
    
    outer_path = None
    inner_paths = []
    
    for dp in data_paths:
        basename = os.path.basename(dp)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(dp)
        else:
            outer_path = dp
    
    # Also scan the directory for inner data files
    if os.path.isdir(std_data_dir):
        for f in os.listdir(std_data_dir):
            full = os.path.join(std_data_dir, f)
            if full in data_paths:
                continue
            if 'parent_function_run_inversion' in f or 'parent_run_inversion' in f:
                inner_paths.append(full)
    
    print(f"[TEST] Outer data: {outer_path}")
    print(f"[TEST] Inner data: {inner_paths}")
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[TEST] Loaded outer data. Keys: {list(outer_data.keys()) if isinstance(outer_data, dict) else 'N/A'}")
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[TEST] Number of args: {len(args)}")
    print(f"[TEST] Kwargs keys: {list(kwargs.keys())}")
    
    # Execute the agent function
    try:
        print("[TEST] Running agent run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("[TEST] Agent run_inversion completed successfully.")
    except Exception as e:
        print(f"[ERROR] Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if chained execution
    if len(inner_paths) > 0:
        print("[TEST] Chained execution detected - not expected for this function, proceeding with direct evaluation.")
    
    # Now we have agent_output (dict) and std_output (dict)
    # We need to call evaluate_results for both
    
    # Extract necessary data for evaluate_results
    # From the function signature: evaluate_results(gt_freq_array, recon_result, clean_spectrum, metadata, results_dir)
    # gt_freq_array comes from the result
    # clean_spectrum and metadata need to come from somewhere
    
    # The metadata is the second argument to run_inversion
    noisy_spectrum = args[0] if len(args) > 0 else kwargs.get('noisy_spectrum', None)
    metadata = args[1] if len(args) > 1 else kwargs.get('metadata', None)
    
    if metadata is None:
        print("[ERROR] Could not extract metadata from arguments")
        sys.exit(1)
    
    # Add noisy_spectrum to metadata for evaluate_results
    metadata_for_eval = dict(metadata)
    metadata_for_eval['noisy_spectrum'] = noisy_spectrum
    
    # We need clean_spectrum - check if it's in metadata
    clean_spectrum = metadata.get('clean_spectrum', None)
    
    # If clean_spectrum is not in metadata, try to generate it from gt_modes and bg_params
    if clean_spectrum is None:
        print("[WARN] clean_spectrum not found in metadata, attempting to reconstruct from forward_operator...")
        try:
            from agent_run_inversion import forward_operator
            gt_modes = metadata['gt_modes']
            freqs = metadata['freqs']
            bg_params = metadata['bg_params']
            linewidth = metadata['linewidth']
            
            mode_params_gt = []
            for (n, l), freq in gt_modes.items():
                # Estimate height - use a reasonable default
                mode_params_gt.append((freq, 1.0, linewidth))
            
            clean_spectrum = forward_operator(mode_params_gt, freqs, bg_params)
            print("[WARN] Reconstructed clean_spectrum with default heights=1.0")
        except Exception as e:
            print(f"[WARN] Could not reconstruct clean_spectrum: {e}")
            # Use smoothed spectrum as proxy
            clean_spectrum = metadata.get('smoothed_spectrum', np.zeros_like(metadata['freqs']))
    
    # Also check if clean_spectrum is stored in the outer_data at top level
    if clean_spectrum is None and 'clean_spectrum' in outer_data:
        clean_spectrum = outer_data['clean_spectrum']
    
    # Get gt_freq_array from results
    agent_gt_freq_array = agent_output.get('gt_freq_array', None)
    std_gt_freq_array = std_output.get('gt_freq_array', None) if std_output else None
    
    gt_freq_array = agent_gt_freq_array if agent_gt_freq_array is not None else std_gt_freq_array
    
    if gt_freq_array is None:
        # Reconstruct from metadata
        gt_modes = metadata['gt_modes']
        gt_freq_array = np.array(sorted(gt_modes.values()))
    
    # Ensure freq_min and freq_max are in metadata
    if 'freq_min' not in metadata_for_eval:
        metadata_for_eval['freq_min'] = metadata_for_eval['freqs'].min()
    if 'freq_max' not in metadata_for_eval:
        metadata_for_eval['freq_max'] = metadata_for_eval['freqs'].max()
    
    # Run evaluate_results for agent
    results_dir_agent = '/tmp/test_results_agent'
    results_dir_std = '/tmp/test_results_std'
    
    try:
        print("\n" + "="*60)
        print("[TEST] Evaluating AGENT output...")
        print("="*60)
        metrics_agent = evaluate_results(gt_freq_array, agent_output, clean_spectrum, metadata_for_eval, results_dir_agent)
    except Exception as e:
        print(f"[ERROR] evaluate_results failed for agent output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        print("\n" + "="*60)
        print("[TEST] Evaluating STANDARD output...")
        print("="*60)
        metrics_std = evaluate_results(gt_freq_array, std_output, clean_spectrum, metadata_for_eval, results_dir_std)
    except Exception as e:
        print(f"[ERROR] evaluate_results failed for standard output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compute composite scores
    score_agent = compute_composite_score(metrics_agent)
    score_std = compute_composite_score(metrics_std)
    
    print("\n" + "="*60)
    print("[RESULTS] Comparison Summary")
    print("="*60)
    print(f"Scores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    
    # Print key metrics side by side
    all_keys = set(list(metrics_agent.keys()) + list(metrics_std.keys()))
    for key in sorted(all_keys):
        agent_val = metrics_agent.get(key, 'N/A')
        std_val = metrics_std.get(key, 'N/A')
        if isinstance(agent_val, float) and isinstance(std_val, float):
            print(f"  {key}: Agent={agent_val:.6f}, Std={std_val:.6f}")
        else:
            print(f"  {key}: Agent={agent_val}, Std={std_val}")
    
    # Determine pass/fail
    # Higher composite score is better
    # Allow 10% margin
    threshold = score_std * 0.90 if score_std > 0 else score_std * 1.10
    
    print(f"\n[TEST] Agent score: {score_agent:.4f}")
    print(f"[TEST] Standard score: {score_std:.4f}")
    print(f"[TEST] Threshold (90% of std): {threshold:.4f}")
    
    # Also check individual critical metrics
    passed = True
    reasons = []
    
    # Check frequency correlation
    agent_cc = metrics_agent.get('frequency_CC', 0)
    std_cc = metrics_std.get('frequency_CC', 0)
    if std_cc > 0 and agent_cc < std_cc * 0.90:
        reasons.append(f"frequency_CC degraded: {agent_cc:.6f} < {std_cc * 0.90:.6f}")
        passed = False
    
    # Check detection rate
    agent_det = metrics_agent.get('detection_rate', 0)
    std_det = metrics_std.get('detection_rate', 0)
    if std_det > 0 and agent_det < std_det * 0.80:
        reasons.append(f"detection_rate degraded: {agent_det:.4f} < {std_det * 0.80:.4f}")
        passed = False
    
    # Check mean relative error (lower is better)
    agent_re = metrics_agent.get('mean_relative_error_pct', float('inf'))
    std_re = metrics_std.get('mean_relative_error_pct', float('inf'))
    if std_re < float('inf') and agent_re > std_re * 2.0:
        reasons.append(f"mean_relative_error_pct degraded: {agent_re:.6f} > {std_re * 2.0:.6f}")
        passed = False
    
    # Check composite score
    if score_agent < threshold:
        reasons.append(f"Composite score too low: {score_agent:.4f} < {threshold:.4f}")
        passed = False
    
    if passed:
        print("\n[PASS] Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("\n[FAIL] Agent performance degraded:")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == '__main__':
    main()