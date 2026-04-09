#!/usr/bin/env python3
"""
Task 155: suite2p_spike — Calcium Spike Deconvolution

Inverse Problem: Recover neural spike times from calcium fluorescence traces.
Forward model: spikes s(t) -> convolve with exponential calcium kernel (tau~1s) -> F(t) + noise
Inverse: From noisy calcium trace F(t) -> estimate spike train s(t) via OASIS algorithm.

OASIS: Online Active Set method for Spike Inference (Friedrich, Zhou & Paninski, 2017)
Implementation follows suite2p's dcnv module.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy.ndimage import gaussian_filter1d


# ============================================================
# OASIS Deconvolution (pure numpy, no numba dependency)
# ============================================================

def oasis_trace(F, tau, fs):
    """
    OASIS: Online Active Set method for Spike Inference on a single trace.
    
    Implements the pool-adjacent-violators algorithm for non-negative
    deconvolution of calcium signals with AR(1) model: c_t = g * c_{t-1} + s_t.
    
    Parameters
    ----------
    F : np.ndarray, shape (T,)
        Fluorescence trace (baseline-corrected).
    tau : float
        Decay time constant in seconds.
    fs : float
        Sampling rate in Hz.
    
    Returns
    -------
    c : np.ndarray, shape (T,)
        Deconvolved calcium trace.
    s : np.ndarray, shape (T,)
        Estimated spike train (non-negative).
    """
    g = np.exp(-1.0 / (tau * fs))  # decay factor per frame
    NT = len(F)
    
    # Pool data structures
    v = np.zeros(NT, dtype=np.float64)  # pool values
    w = np.zeros(NT, dtype=np.float64)  # pool weights
    t = np.zeros(NT, dtype=np.int64)    # pool start times
    l = np.zeros(NT, dtype=np.int64)    # pool lengths
    
    it = 0  # frame index
    ip = 0  # pool index
    
    while it < NT:
        v[ip] = F[it]
        w[ip] = 1.0
        t[ip] = it
        l[ip] = 1
        
        while ip > 0:
            if v[ip - 1] * (g ** l[ip - 1]) > v[ip]:
                # Violation: merge pools
                f1 = g ** l[ip - 1]
                f2 = g ** (2 * l[ip - 1])
                wnew = w[ip - 1] + w[ip] * f2
                v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                w[ip - 1] = wnew
                l[ip - 1] = l[ip - 1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1
    
    # Extract spikes from pools
    s = np.zeros(NT, dtype=np.float64)
    n_pools = ip
    for i in range(1, n_pools):
        spike_val = v[i] - v[i - 1] * (g ** l[i - 1])
        if spike_val > 0:
            s[t[i]] = spike_val
    # First pool
    if n_pools > 0 and v[0] > 0:
        s[t[0]] = v[0]
    
    # Reconstruct calcium from spikes
    c = np.zeros(NT, dtype=np.float64)
    c[0] = s[0]
    for i in range(1, NT):
        c[i] = g * c[i - 1] + s[i]
    
    return c, s


def oasis_batch(F, tau, fs):
    """
    Run OASIS on multiple traces.
    
    Parameters
    ----------
    F : np.ndarray, shape (N, T)
        Fluorescence traces for N neurons.
    tau : float
        Decay time constant in seconds.
    fs : float
        Sampling rate in Hz.
    
    Returns
    -------
    C : np.ndarray, shape (N, T)
        Deconvolved calcium traces.
    S : np.ndarray, shape (N, T)
        Estimated spike trains.
    """
    N, T = F.shape
    C = np.zeros_like(F)
    S = np.zeros_like(F)
    for i in range(N):
        C[i], S[i] = oasis_trace(F[i], tau, fs)
    return C, S


# ============================================================
# Forward Model: spikes -> calcium fluorescence
# ============================================================

def generate_spike_train(n_frames, spike_rate, fs, rng):
    """
    Generate a Poisson spike train.
    
    Parameters
    ----------
    n_frames : int
        Number of time frames.
    spike_rate : float
        Average spike rate in Hz.
    fs : float
        Sampling rate in Hz.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    spikes : np.ndarray, shape (n_frames,)
        Binary spike train (0 or 1 per frame).
    """
    prob_per_frame = spike_rate / fs
    spikes = (rng.random(n_frames) < prob_per_frame).astype(np.float64)
    return spikes


def forward_calcium_model(spikes, tau, fs, amplitude=1.0, baseline=1.0, noise_std=0.1, rng=None):
    """
    Forward model: convolve spike train with exponential kernel, add noise.
    
    F(t) = baseline + sum_i amplitude * exp(-(t - t_i) / tau) + noise
    
    Equivalently in AR(1) form: c_t = g * c_{t-1} + a * s_t, F_t = baseline + c_t + noise
    
    Parameters
    ----------
    spikes : np.ndarray, shape (T,)
        Ground truth spike train.
    tau : float
        Calcium decay time constant in seconds.
    fs : float
        Sampling rate in Hz.
    amplitude : float
        Spike amplitude (calcium transient size).
    baseline : float
        Baseline fluorescence F0.
    noise_std : float
        Standard deviation of Gaussian noise.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    F : np.ndarray, shape (T,)
        Noisy calcium fluorescence trace.
    clean_calcium : np.ndarray, shape (T,)
        Noise-free calcium component.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    g = np.exp(-1.0 / (tau * fs))
    T = len(spikes)
    
    # AR(1) convolution
    clean_calcium = np.zeros(T, dtype=np.float64)
    clean_calcium[0] = amplitude * spikes[0]
    for t in range(1, T):
        clean_calcium[t] = g * clean_calcium[t - 1] + amplitude * spikes[t]
    
    # Add Gaussian noise
    noise = rng.normal(0, noise_std, T)
    
    # Total fluorescence
    F = baseline + clean_calcium + noise
    
    return F, clean_calcium


# ============================================================
# Evaluation Metrics
# ============================================================

def detect_spikes_from_deconv(s, threshold_factor=0.3):
    """
    Detect discrete spike events from deconvolved trace.
    
    Parameters
    ----------
    s : np.ndarray
        Deconvolved spike magnitude trace.
    threshold_factor : float
        Fraction of max spike amplitude used as detection threshold.
    
    Returns
    -------
    spike_frames : np.ndarray
        Frame indices of detected spikes.
    """
    if np.max(s) <= 0:
        return np.array([], dtype=int)
    
    threshold = threshold_factor * np.max(s)
    # Also use a minimum absolute threshold
    abs_threshold = np.percentile(s[s > 0], 50) if np.any(s > 0) else 0
    threshold = max(threshold, abs_threshold)
    
    spike_frames = np.where(s > threshold)[0]
    return spike_frames


def evaluate_spike_detection(true_spikes, detected_frames, tolerance=3):
    """
    Evaluate spike detection accuracy with temporal tolerance.
    
    Parameters
    ----------
    true_spikes : np.ndarray
        Ground truth spike train (binary).
    detected_frames : np.ndarray
        Detected spike frame indices.
    tolerance : int
        Temporal tolerance in frames for matching.
    
    Returns
    -------
    metrics : dict
        precision, recall, F1, temporal_error (mean frames)
    """
    true_frames = np.where(true_spikes > 0)[0]
    
    if len(true_frames) == 0 and len(detected_frames) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'temporal_error_frames': 0.0}
    if len(true_frames) == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'temporal_error_frames': float('nan')}
    if len(detected_frames) == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0, 'temporal_error_frames': float('nan')}
    
    # Match detected to true (greedy, within tolerance)
    matched_true = set()
    matched_det = set()
    temporal_errors = []
    
    # Sort by distance to nearest true spike
    for d in detected_frames:
        dists = np.abs(true_frames - d)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]
        
        if nearest_dist <= tolerance and nearest_idx not in matched_true:
            matched_true.add(nearest_idx)
            matched_det.add(d)
            temporal_errors.append(nearest_dist)
    
    tp = len(matched_true)
    fp = len(detected_frames) - tp
    fn = len(true_frames) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mean_temporal_error = np.mean(temporal_errors) if temporal_errors else float('nan')
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'temporal_error_frames': float(mean_temporal_error),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }


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


# ============================================================
# Main Pipeline
# ============================================================

def main():
    # ---- Parameters ----
    fs = 30.0           # sampling rate (Hz)
    dt = 1.0 / fs       # frame interval (seconds)
    duration = 60.0     # seconds
    n_frames = int(duration * fs)  # 1800 frames
    n_neurons = 5
    spike_rate = 2.0    # Hz per neuron
    tau = 1.0           # calcium decay time constant (seconds)
    amplitude = 1.0     # spike amplitude
    baseline = 1.0      # baseline fluorescence F0
    noise_std = 0.08    # noise level (SNR ~ 10-15)
    
    rng = np.random.default_rng(42)
    
    # ---- Output directory ----
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # ---- Generate synthetic data ----
    print(f"Generating {n_neurons} neurons, {n_frames} frames ({duration}s at {fs} Hz)")
    print(f"Spike rate: {spike_rate} Hz, tau: {tau}s, amplitude: {amplitude}, noise_std: {noise_std}")
    
    true_spikes = np.zeros((n_neurons, n_frames), dtype=np.float64)
    calcium_traces = np.zeros((n_neurons, n_frames), dtype=np.float64)
    clean_calcium = np.zeros((n_neurons, n_frames), dtype=np.float64)
    
    for i in range(n_neurons):
        true_spikes[i] = generate_spike_train(n_frames, spike_rate, fs, rng)
        calcium_traces[i], clean_calcium[i] = forward_calcium_model(
            true_spikes[i], tau, fs, amplitude, baseline, noise_std, rng
        )
        n_spk = int(np.sum(true_spikes[i]))
        print(f"  Neuron {i}: {n_spk} spikes")
    
    # ---- Preprocess: baseline correction ----
    # Subtract running minimum (simple baseline correction)
    F_corrected = np.zeros_like(calcium_traces)
    for i in range(n_neurons):
        # Smooth then compute rolling minimum
        smoothed = gaussian_filter1d(calcium_traces[i], sigma=fs * 0.1)
        win = int(fs * 30)  # 30-second window
        baseline_est = np.array([
            np.min(smoothed[max(0, j - win):j + win + 1])
            for j in range(n_frames)
        ])
        F_corrected[i] = calcium_traces[i] - baseline_est
        # Clip negative values
        F_corrected[i] = np.maximum(F_corrected[i], 0)
    
    # ---- Run OASIS deconvolution ----
    print("\nRunning OASIS deconvolution...")
    C_deconv, S_deconv = oasis_batch(F_corrected, tau, fs)
    print("Deconvolution complete.")
    
    # ---- Evaluate each neuron ----
    all_metrics = []
    for i in range(n_neurons):
        detected = detect_spikes_from_deconv(S_deconv[i], threshold_factor=0.2)
        det_metrics = evaluate_spike_detection(true_spikes[i], detected, tolerance=3)
        
        # Correlation between deconvolved and true spike train
        # Smooth both for correlation (since point spikes are hard to match exactly)
        smooth_true = gaussian_filter1d(true_spikes[i], sigma=2.0)
        smooth_deconv = gaussian_filter1d(S_deconv[i], sigma=2.0)
        cc = compute_correlation(smooth_true, smooth_deconv)
        
        # PSNR of deconvolved calcium vs clean calcium
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
        print(f"  Neuron {i}: F1={det_metrics['f1']:.3f}, CC={cc:.3f}, PSNR={psnr:.2f} dB, "
              f"True={int(np.sum(true_spikes[i]))}, Detected={len(detected)}")
    
    # ---- Aggregate metrics ----
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_cc = np.mean([m['correlation'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr_dB'] for m in all_metrics])
    avg_temporal_error = np.nanmean([m['temporal_error_frames'] for m in all_metrics])
    
    summary = {
        'task': 'suite2p_spike',
        'method': 'OASIS (Online Active Set for Spike Inference)',
        'n_neurons': n_neurons,
        'n_frames': n_frames,
        'sampling_rate_Hz': fs,
        'duration_s': duration,
        'tau_s': tau,
        'spike_rate_Hz': spike_rate,
        'noise_std': noise_std,
        'avg_precision': round(avg_precision, 4),
        'avg_recall': round(avg_recall, 4),
        'avg_f1': round(avg_f1, 4),
        'avg_correlation': round(avg_cc, 4),
        'avg_psnr_dB': round(avg_psnr, 2),
        'avg_temporal_error_frames': round(avg_temporal_error, 2),
        'per_neuron': all_metrics,
    }
    
    print(f"\n=== Summary ===")
    print(f"  Avg F1:         {avg_f1:.4f}")
    print(f"  Avg Precision:  {avg_precision:.4f}")
    print(f"  Avg Recall:     {avg_recall:.4f}")
    print(f"  Avg CC:         {avg_cc:.4f}")
    print(f"  Avg PSNR:       {avg_psnr:.2f} dB")
    print(f"  Avg Temp Error: {avg_temporal_error:.2f} frames")
    
    # ---- Save outputs ----
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save ground truth and reconstruction
    gt_path = os.path.join(results_dir, 'ground_truth.npy')
    recon_path = os.path.join(results_dir, 'reconstruction.npy')
    np.save(gt_path, true_spikes)
    np.save(recon_path, S_deconv)
    print(f"Saved ground_truth.npy and reconstruction.npy")
    
    # Also save calcium traces
    np.save(os.path.join(results_dir, 'calcium_traces.npy'), calcium_traces)
    np.save(os.path.join(results_dir, 'clean_calcium.npy'), clean_calcium)
    
    # ---- Visualization ----
    print("\nGenerating visualization...")
    time_axis = np.arange(n_frames) / fs  # time in seconds
    
    # Show first 3 neurons, first 20 seconds
    show_neurons = min(3, n_neurons)
    show_frames = int(20 * fs)  # 20 seconds
    t_show = time_axis[:show_frames]
    
    fig, axes = plt.subplots(4, show_neurons, figsize=(6 * show_neurons, 12),
                             sharex='col', squeeze=False)
    
    for ni in range(show_neurons):
        # Panel 1: True spike train
        ax = axes[0, ni]
        spike_times = np.where(true_spikes[ni, :show_frames] > 0)[0] / fs
        ax.vlines(spike_times, 0, 1, colors='black', linewidth=0.8)
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Neuron {ni}: True Spikes ({int(np.sum(true_spikes[ni]))} total)')
        if ni == 0:
            ax.annotate('(a) Ground Truth', xy=(0.02, 0.85), xycoords='axes fraction',
                        fontsize=9, fontweight='bold', color='darkblue')
        
        # Panel 2: Calcium trace F(t)
        ax = axes[1, ni]
        ax.plot(t_show, calcium_traces[ni, :show_frames], 'b-', linewidth=0.5, alpha=0.8, label='Noisy F(t)')
        ax.plot(t_show, baseline + clean_calcium[ni, :show_frames], 'r-', linewidth=0.8, alpha=0.6, label='Clean F(t)')
        ax.set_ylabel('Fluorescence')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(f'Calcium Trace (SNR≈{amplitude / noise_std:.1f})')
        if ni == 0:
            ax.annotate('(b) Observation', xy=(0.02, 0.85), xycoords='axes fraction',
                        fontsize=9, fontweight='bold', color='darkblue')
        
        # Panel 3: Deconvolved spikes
        ax = axes[2, ni]
        ax.plot(t_show, S_deconv[ni, :show_frames], 'g-', linewidth=0.8)
        detected = detect_spikes_from_deconv(S_deconv[ni], threshold_factor=0.2)
        detected_in_window = detected[detected < show_frames]
        if len(detected_in_window) > 0:
            ax.vlines(detected_in_window / fs, 0, np.max(S_deconv[ni, :show_frames]) * 0.3,
                      colors='red', linewidth=0.8, alpha=0.5, label='Detected')
        ax.set_ylabel('Spike Magnitude')
        ax.legend(fontsize=7, loc='upper right')
        nm = all_metrics[ni]
        ax.set_title(f'Deconvolved (F1={nm["f1"]:.2f}, CC={nm["correlation"]:.2f})')
        if ni == 0:
            ax.annotate('(c) OASIS Deconvolution', xy=(0.02, 0.85), xycoords='axes fraction',
                        fontsize=9, fontweight='bold', color='darkblue')
        
        # Panel 4: Overlay comparison
        ax = axes[3, ni]
        spike_times_true = np.where(true_spikes[ni, :show_frames] > 0)[0] / fs
        ax.vlines(spike_times_true, 0, 1, colors='black', linewidth=1.0, alpha=0.7, label='True')
        if len(detected_in_window) > 0:
            ax.vlines(detected_in_window / fs, 0, 0.8, colors='red', linewidth=1.0, alpha=0.5, label='Detected')
        ax.set_ylim(-0.1, 1.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spikes')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(f'Overlay (P={nm["precision"]:.2f}, R={nm["recall"]:.2f})')
        if ni == 0:
            ax.annotate('(d) Comparison', xy=(0.02, 0.85), xycoords='axes fraction',
                        fontsize=9, fontweight='bold', color='darkblue')
    
    fig.suptitle('Calcium Spike Deconvolution (OASIS)\n'
                 f'Avg F1={avg_f1:.3f}, CC={avg_cc:.3f}, PSNR={avg_psnr:.1f} dB',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {vis_path}")
    
    print("\n=== Task 155 suite2p_spike COMPLETE ===")
    return summary


if __name__ == '__main__':
    main()
