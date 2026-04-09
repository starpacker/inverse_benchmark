import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import json
from scipy.ndimage import gaussian_filter1d

# Inject the referee function (evaluate_results) from Reference B
def detect_spikes_from_deconv(s, threshold_factor=0.3):
    """
    Detect discrete spike events from deconvolved trace.
    """
    if np.max(s) <= 0:
        return np.array([], dtype=int)
    
    threshold = threshold_factor * np.max(s)
    abs_threshold = np.percentile(s[s > 0], 50) if np.any(s > 0) else 0
    threshold = max(threshold, abs_threshold)
    
    spike_frames = np.where(s > threshold)[0]
    return spike_frames

def compute_correlation(x, y):
    """Pearson correlation coefficient."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def compute_psnr(clean, noisy):
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((clean - noisy) ** 2)
    if mse == 0:
        return float('inf')
    peak = np.max(np.abs(clean))
    if peak == 0:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))

def evaluate_results(true_spikes, S_deconv, clean_calcium, C_deconv, params, results_dir=None):
    """
    Evaluate spike detection accuracy and reconstruction quality.
    
    Parameters
    ----------
    true_spikes : np.ndarray, shape (N, T)
        Ground truth spike trains.
    S_deconv : np.ndarray, shape (N, T)
        Estimated spike trains from OASIS.
    clean_calcium : np.ndarray, shape (N, T)
        Noise-free calcium traces.
    C_deconv : np.ndarray, shape (N, T)
        Deconvolved calcium traces.
    params : dict
        Simulation parameters.
    results_dir : str or None
        Directory to save results. If None, results are not saved.
    
    Returns
    -------
    summary : dict
        Summary metrics and per-neuron results.
    """
    n_neurons = true_spikes.shape[0]
    n_frames = true_spikes.shape[1]
    fs = params['fs']
    tolerance = 3
    
    all_metrics = []
    
    for i in range(n_neurons):
        detected = detect_spikes_from_deconv(S_deconv[i], threshold_factor=0.2)
        
        # Evaluate spike detection
        true_frames = np.where(true_spikes[i] > 0)[0]
        
        if len(true_frames) == 0 and len(detected) == 0:
            det_metrics = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'temporal_error_frames': 0.0,
                          'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        elif len(true_frames) == 0:
            det_metrics = {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'temporal_error_frames': float('nan'),
                          'true_positives': 0, 'false_positives': len(detected), 'false_negatives': 0}
        elif len(detected) == 0:
            det_metrics = {'precision': 1.0, 'recall': 0.0, 'f1': 0.0, 'temporal_error_frames': float('nan'),
                          'true_positives': 0, 'false_positives': 0, 'false_negatives': len(true_frames)}
        else:
            matched_true = set()
            matched_det = set()
            temporal_errors = []
            
            for d in detected:
                dists = np.abs(true_frames - d)
                nearest_idx = np.argmin(dists)
                nearest_dist = dists[nearest_idx]
                
                if nearest_dist <= tolerance and nearest_idx not in matched_true:
                    matched_true.add(nearest_idx)
                    matched_det.add(d)
                    temporal_errors.append(nearest_dist)
            
            tp = len(matched_true)
            fp = len(detected) - tp
            fn = len(true_frames) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            mean_temporal_error = np.mean(temporal_errors) if temporal_errors else float('nan')
            
            det_metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'temporal_error_frames': float(mean_temporal_error),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
            }
        
        # Correlation
        smooth_true = gaussian_filter1d(true_spikes[i], sigma=2.0)
        smooth_deconv = gaussian_filter1d(S_deconv[i], sigma=2.0)
        cc = compute_correlation(smooth_true, smooth_deconv)
        
        # PSNR
        psnr = compute_psnr(clean_calcium[i], C_deconv[i])
        
        neuron_metrics = {
            'neuron': i,
            **det_metrics,
            'correlation': cc,
            'psnr_dB': psnr,
            'n_true_spikes': int(np.sum(true_spikes[i])),
            'n_detected_spikes': len(detected),
        }
        all_metrics.append(neuron_metrics)
    
    # Aggregate metrics
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_cc = np.mean([m['correlation'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr_dB'] for m in all_metrics])
    avg_temporal_error = np.nanmean([m['temporal_error_frames'] for m in all_metrics])
    
    duration = n_frames / fs
    
    summary = {
        'task': 'suite2p_spike',
        'method': 'OASIS (Online Active Set for Spike Inference)',
        'n_neurons': n_neurons,
        'n_frames': n_frames,
        'sampling_rate_Hz': fs,
        'duration_s': duration,
        'tau_s': params['tau'],
        'spike_rate_Hz': params['spike_rate'],
        'noise_std': params['noise_std'],
        'avg_precision': round(avg_precision, 4),
        'avg_recall': round(avg_recall, 4),
        'avg_f1': round(avg_f1, 4),
        'avg_correlation': round(avg_cc, 4),
        'avg_psnr_dB': round(avg_psnr, 2),
        'avg_temporal_error_frames': round(avg_temporal_error, 2),
        'per_neuron': all_metrics,
    }
    
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        
        metrics_path = os.path.join(results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        np.save(os.path.join(results_dir, 'ground_truth.npy'), true_spikes)
        np.save(os.path.join(results_dir, 'reconstruction.npy'), S_deconv)
    
    return summary


def main():
    # Data paths provided
    data_paths = ['/data/yjh/suite2p_spike_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze data paths to determine execution pattern
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No primary data file found.")
        sys.exit(1)
    
    # Load outer (primary) data
    print(f"Loading primary data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {outer_data.get('func_name', 'unknown')}")
    print(f"Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Execute the agent function
    print("Running agent function: run_inversion")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution)
    if inner_data_paths:
        print(f"Chained execution detected. Inner data files: {inner_data_paths}")
        # Load inner data
        inner_data_path = inner_data_paths[0]
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute returned operator
        if callable(agent_output):
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing returned operator: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("ERROR: Expected agent_output to be callable for chained execution")
            sys.exit(1)
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    print("Agent execution completed successfully.")
    
    # For evaluation, we need ground truth data
    # The evaluate_results function requires: true_spikes, S_deconv, clean_calcium, C_deconv, params
    # Since we only have the function inputs and outputs, we need to construct evaluation data
    
    # Extract what we have from the input args
    # Based on the function signature: run_inversion(F_corrected, tau, fs)
    # args[0] = F_corrected, args[1] = tau (or kwargs), args[2] = fs (or kwargs)
    
    F_corrected = args[0] if len(args) > 0 else kwargs.get('F_corrected', None)
    tau = args[1] if len(args) > 1 else kwargs.get('tau', 0.7)
    fs = args[2] if len(args) > 2 else kwargs.get('fs', 30.0)
    
    # Extract results
    agent_C_deconv = final_result.get('C_deconv', None)
    agent_S_deconv = final_result.get('S_deconv', None)
    
    std_C_deconv = std_result.get('C_deconv', None) if isinstance(std_result, dict) else None
    std_S_deconv = std_result.get('S_deconv', None) if isinstance(std_result, dict) else None
    
    if agent_S_deconv is None or std_S_deconv is None:
        print("ERROR: Missing deconvolved results")
        sys.exit(1)
    
    print(f"Agent S_deconv shape: {agent_S_deconv.shape}")
    print(f"Standard S_deconv shape: {std_S_deconv.shape}")
    
    # Since we don't have ground truth spikes and clean calcium from the pkl file,
    # we'll compare the agent output directly with the standard output
    # Using correlation and MSE as metrics
    
    # Compute correlation between agent and standard outputs
    def compute_output_similarity(agent_arr, std_arr):
        """Compute similarity metrics between agent and standard outputs."""
        if agent_arr.shape != std_arr.shape:
            print(f"Shape mismatch: agent {agent_arr.shape} vs std {std_arr.shape}")
            return 0.0, float('inf')
        
        # Flatten for overall correlation
        agent_flat = agent_arr.flatten()
        std_flat = std_arr.flatten()
        
        # Correlation
        if np.std(agent_flat) > 0 and np.std(std_flat) > 0:
            corr = np.corrcoef(agent_flat, std_flat)[0, 1]
        else:
            corr = 1.0 if np.allclose(agent_flat, std_flat) else 0.0
        
        # MSE
        mse = np.mean((agent_flat - std_flat) ** 2)
        
        return corr, mse
    
    # Evaluate S_deconv similarity
    s_corr, s_mse = compute_output_similarity(agent_S_deconv, std_S_deconv)
    print(f"S_deconv - Correlation: {s_corr:.6f}, MSE: {s_mse:.6e}")
    
    # Evaluate C_deconv similarity
    c_corr, c_mse = compute_output_similarity(agent_C_deconv, std_C_deconv)
    print(f"C_deconv - Correlation: {c_corr:.6f}, MSE: {c_mse:.6e}")
    
    # Overall score (average correlation, higher is better)
    score_agent = (s_corr + c_corr) / 2.0
    score_std = 1.0  # Standard is the reference (perfect match)
    
    print(f"\nScores -> Agent: {score_agent:.6f}, Standard: {score_std:.6f}")
    
    # Additional check: verify the outputs are numerically close
    s_close = np.allclose(agent_S_deconv, std_S_deconv, rtol=1e-5, atol=1e-8)
    c_close = np.allclose(agent_C_deconv, std_C_deconv, rtol=1e-5, atol=1e-8)
    
    print(f"S_deconv numerically close: {s_close}")
    print(f"C_deconv numerically close: {c_close}")
    
    # Determine success
    # We allow a margin of error of 10% for correlation
    # Correlation should be at least 0.9 (90% of perfect)
    threshold = 0.9
    
    if score_agent >= threshold:
        print(f"\nSUCCESS: Agent performance is acceptable (score >= {threshold})")
        sys.exit(0)
    else:
        print(f"\nFAILURE: Agent performance degraded (score < {threshold})")
        sys.exit(1)


if __name__ == "__main__":
    main()