import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def compute_si_sdr(ref, est):
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e_noise = est - s_target
    return 10.0 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-12))

def compute_psnr(ref, est):
    """Peak SNR (dB) for 1-D signal."""
    mse = np.mean((ref - est) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(np.abs(ref))
    return 10.0 * np.log10(peak ** 2 / mse)

def compute_cc(ref, est):
    """Pearson correlation coefficient."""
    return float(np.corrcoef(ref, est)[0, 1])

def evaluate_results(sources, recovered_scaled, t, mixed, params, results_dir):
    """
    Evaluate reconstruction quality and save metrics, arrays, and visualizations.
    
    Parameters
    ----------
    sources : np.ndarray
        Ground truth sources, shape (n_sources, N).
    recovered_scaled : np.ndarray
        Recovered and rescaled sources, shape (n_sources, N).
    t : np.ndarray
        Time array, shape (N,).
    mixed : np.ndarray
        Mixed observations, shape (n_sensors, N).
    params : dict
        Dictionary with sample rate, duration, etc.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    n_sources = sources.shape[0]
    sr = params['sr']
    duration = params['duration']
    
    # Compute metrics for each source
    si_sdr_vals = [compute_si_sdr(sources[i], recovered_scaled[i]) for i in range(n_sources)]
    psnr_vals = [compute_psnr(sources[i], recovered_scaled[i]) for i in range(n_sources)]
    cc_vals = [compute_cc(sources[i], recovered_scaled[i]) for i in range(n_sources)]
    
    avg_si_sdr = float(np.mean(si_sdr_vals))
    avg_psnr = float(np.mean(psnr_vals))
    avg_cc = float(np.mean(cc_vals))
    
    metrics = {
        "si_sdr_db": round(avg_si_sdr, 4),
        "si_sdr_per_source_db": [round(v, 4) for v in si_sdr_vals],
        "psnr_db": round(avg_psnr, 4),
        "psnr_per_source_db": [round(v, 4) for v in psnr_vals],
        "correlation_coefficient": round(avg_cc, 6),
        "cc_per_source": [round(v, 6) for v in cc_vals],
        "n_sources": n_sources,
        "sample_rate": sr,
        "duration_s": duration,
        "method": "FastICA (sklearn)",
    }
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[METRICS] SI-SDR (avg) = {avg_si_sdr:.2f} dB  ({si_sdr_vals})")
    print(f"[METRICS] PSNR  (avg) = {avg_psnr:.2f} dB  ({psnr_vals})")
    print(f"[METRICS] CC    (avg) = {avg_cc:.6f}  ({cc_vals})")
    print(f"[INFO] Saved metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), sources)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recovered_scaled)
    print("[INFO] Saved ground_truth.npy and reconstruction.npy")
    
    # Visualization
    T_PLOT = 0.05  # show first 50 ms for clarity
    n_plot = int(sr * T_PLOT)
    t_plot = t[:n_plot] * 1000  # ms
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    
    titles_src = ["GT Source 1 (440 + 880 Hz)", "GT Source 2 (330 + 660 Hz + saw)"]
    titles_mix = [f"Mixed Signal {i+1} (mic {i+1})" for i in range(mixed.shape[0])]
    titles_rec = ["Recovered Source 1", "Recovered Source 2"]
    titles_res = ["Residual |GT − Rec| Source 1", "Residual |GT − Rec| Source 2"]
    
    for j in range(2):
        # Row 0 — GT sources
        axes[0, j].plot(t_plot, sources[j, :n_plot], color="tab:blue", lw=0.8)
        axes[0, j].set_title(titles_src[j], fontsize=10)
        axes[0, j].set_ylabel("Amplitude")
        
        # Row 1 — Mixed (show first 2 of n_mix sensors)
        axes[1, j].plot(t_plot, mixed[j, :n_plot], color="tab:orange", lw=0.8)
        axes[1, j].set_title(titles_mix[j], fontsize=10)
        axes[1, j].set_ylabel("Amplitude")
        
        # Row 2 — Recovered
        axes[2, j].plot(t_plot, recovered_scaled[j, :n_plot], color="tab:green", lw=0.8)
        axes[2, j].set_title(titles_rec[j], fontsize=10)
        axes[2, j].set_ylabel("Amplitude")
        
        # Row 3 — Residual
        residual = np.abs(sources[j, :n_plot] - recovered_scaled[j, :n_plot])
        axes[3, j].plot(t_plot, residual, color="tab:red", lw=0.8)
        axes[3, j].set_title(titles_res[j], fontsize=10)
        axes[3, j].set_ylabel("|Residual|")
        axes[3, j].set_xlabel("Time (ms)")
    
    # Add metrics text
    metrics_txt = (
        f"SI-SDR: {avg_si_sdr:.2f} dB  |  PSNR: {avg_psnr:.2f} dB  |  CC: {avg_cc:.6f}\n"
        f"Src1 → SI-SDR={si_sdr_vals[0]:.2f}, PSNR={psnr_vals[0]:.2f}, CC={cc_vals[0]:.6f}   "
        f"Src2 → SI-SDR={si_sdr_vals[1]:.2f}, PSNR={psnr_vals[1]:.2f}, CC={cc_vals[1]:.6f}"
    )
    fig.suptitle(
        "Task 175: Audio Blind Source Separation (FastICA)\n" + metrics_txt,
        fontsize=12, fontweight="bold", y=1.02,
    )
    
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualisation → {vis_path}")
    
    return metrics
