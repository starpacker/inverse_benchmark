import matplotlib

matplotlib.use('Agg')

import numpy as np

import json

import os

from scipy.ndimage import gaussian_filter1d

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
