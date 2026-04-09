import os

import json

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.fft import irfft

def evaluate_results(
    data_dict,
    inversion_result,
    injection_parameters,
    duration,
    sampling_freq,
    minimum_freq,
    outdir,
):
    """
    Evaluate and visualize the inversion results.
    
    Computes metrics (PSNR, correlation, waveform match), saves outputs,
    and creates visualization plots.
    
    Args:
        data_dict: output from load_and_preprocess_data
        inversion_result: output from run_inversion
        injection_parameters: true injection parameters
        duration: signal duration in seconds
        sampling_freq: sampling frequency in Hz
        minimum_freq: minimum frequency cutoff
        outdir: output directory
    
    Returns:
        dict of computed metrics
    """
    gt_strain_fd = data_dict["gt_strain_fd"]
    noisy_strain_fd = data_dict["noisy_strain_fd"]
    freq_array = data_dict["freq_array"]
    h1 = data_dict["h1"]
    ifos = data_dict["ifos"]
    
    recon_signal_fd = inversion_result["recon_signal_fd"]
    relative_errors = inversion_result["relative_errors"]
    median_params = inversion_result["median_params"]
    true_values = inversion_result["true_values"]
    all_report_params = inversion_result["all_report_params"]
    t_elapsed = inversion_result["runtime"]
    result = inversion_result["result"]
    
    # Signal-level metrics
    mask = (freq_array >= minimum_freq) & (freq_array <= sampling_freq / 2)
    gt_signal = gt_strain_fd[mask]
    recon_signal = recon_signal_fd[mask]
    
    gt_abs = np.abs(gt_signal)
    recon_abs = np.abs(recon_signal)
    mse = np.mean((gt_abs - recon_abs) ** 2)
    max_val = np.max(gt_abs)
    psnr = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float("inf")
    
    corr_complex = np.abs(np.vdot(gt_signal, recon_signal)) / (
        np.linalg.norm(gt_signal) * np.linalg.norm(recon_signal)
    )
    
    def inner_product(a, b, psd, df):
        return 4.0 * df * np.real(np.sum(np.conj(a) * b / psd))
    
    psd_array = h1.power_spectral_density_array[mask]
    df = freq_array[1] - freq_array[0]
    psd_safe = np.where(psd_array > 0, psd_array, np.inf)
    
    overlap = inner_product(gt_signal, recon_signal, psd_safe, df)
    norm_gt = np.sqrt(inner_product(gt_signal, gt_signal, psd_safe, df))
    norm_recon = np.sqrt(inner_product(recon_signal, recon_signal, psd_safe, df))
    match = overlap / (norm_gt * norm_recon) if (norm_gt > 0 and norm_recon > 0) else 0.0
    
    mean_rel_error = np.mean(list(relative_errors.values()))
    
    snr_list = [float(ifo.meta_data.get("optimal_SNR", 0.0)) for ifo in ifos]
    network_snr = np.sqrt(sum(s ** 2 for s in snr_list))
    
    print(f"\n=== Signal Metrics ===")
    print(f"PSNR (freq-domain amplitude): {psnr:.2f} dB")
    print(f"Correlation coefficient:       {corr_complex:.6f}")
    print(f"Waveform match (overlap):      {match:.6f}")
    print(f"Mean param relative error:     {mean_rel_error:.2f}%")
    print(f"Network SNR (injected):        {network_snr:.1f}")
    print(f"Runtime:                       {t_elapsed:.1f} s")
    
    # Save metrics
    metrics = {
        "psnr_db": round(float(psnr), 2),
        "correlation": round(float(corr_complex), 6),
        "waveform_match": round(float(match), 6),
        "mean_relative_error_pct": round(float(mean_rel_error), 2),
        "network_snr": round(float(network_snr), 2),
        "runtime_s": round(float(t_elapsed), 1),
        "nlive": 100,
        "n_free_params": 3,
        "sampler": "dynesty",
        "parameter_errors": {p: round(float(v), 4) for p, v in relative_errors.items()},
        "recovered_parameters": {p: round(float(v), 4) for p, v in median_params.items()},
        "true_parameters": {p: round(float(true_values[p]), 4) for p in all_report_params},
    }
    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save numpy arrays
    print("[7/7] Saving outputs and creating visualization...")
    n_samples = int(duration * sampling_freq)
    gt_td = np.real(irfft(gt_strain_fd, n=n_samples))
    recon_td = np.real(irfft(recon_signal_fd, n=n_samples))
    noisy_td = np.real(irfft(noisy_strain_fd, n=n_samples))
    time_array = np.arange(n_samples) / sampling_freq
    
    np.save(os.path.join(outdir, "ground_truth.npy"), gt_td)
    np.save(os.path.join(outdir, "reconstruction.npy"), recon_td)
    print("Saved ground_truth.npy and reconstruction.npy")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    t_start_data = injection_parameters["geocent_time"] - duration + 2
    t_plot = time_array + t_start_data - injection_parameters["geocent_time"]
    
    ax = axes[0, 0]
    ax.plot(t_plot, noisy_td, color="lightgray", alpha=0.5, linewidth=0.3,
            label="Noisy data", rasterized=True)
    ax.plot(t_plot, gt_td, color="tab:blue", linewidth=1.0, label="True signal")
    ax.set_xlabel("Time relative to merger (s)")
    ax.set_ylabel("Strain")
    ax.set_title("Detector Strain (H1): Data + Injected Signal")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, 0.05)
    
    ax = axes[0, 1]
    ax.plot(t_plot, gt_td, color="tab:blue", linewidth=1.2, label="True signal")
    ax.plot(t_plot, recon_td, color="tab:red", linewidth=1.0, linestyle="--",
            label="Recovered (MAP)")
    ax.set_xlabel("Time relative to merger (s)")
    ax.set_ylabel("Strain")
    ax.set_title(f"Waveform Recovery (match={match:.4f})")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, 0.05)
    
    ax = axes[1, 0]
    param_labels = {
        "chirp_mass": "$\\mathcal{M}$",
        "mass_ratio": "$q$",
        "luminosity_distance": "$d_L$",
        "mass_1": "$m_1$",
        "mass_2": "$m_2$",
    }
    x_pos = np.arange(len(all_report_params))
    rel_errs = [relative_errors[p] for p in all_report_params]
    colors = ["tab:green" if e < 5 else "tab:orange" if e < 15 else "tab:red" for e in rel_errs]
    ax.bar(x_pos, rel_errs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([param_labels.get(p, p) for p in all_report_params], fontsize=11)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Parameter Recovery Accuracy")
    ax.axhline(y=5, color="green", linestyle=":", alpha=0.5, label="5%")
    ax.axhline(y=15, color="orange", linestyle=":", alpha=0.5, label="15%")
    ax.legend()
    
    ax = axes[1, 1]
    freq_plot = freq_array[mask]
    ax.loglog(freq_plot, np.abs(gt_signal), color="tab:blue", linewidth=1.0,
              label="True signal")
    ax.loglog(freq_plot, np.abs(recon_signal), color="tab:red", linewidth=0.8,
              linestyle="--", label="Recovered (MAP)")
    ax.loglog(freq_plot, np.sqrt(psd_safe), color="lightgray", linewidth=0.5,
              alpha=0.8, label="$\\sqrt{S_n(f)}$")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|h(f)|")
    ax.set_title(f"Frequency-Domain Comparison (PSNR={psnr:.1f} dB)")
    ax.legend(loc="upper right")
    ax.set_xlim(minimum_freq, sampling_freq / 2)
    
    plt.suptitle(
        f"Bilby GW Parameter Estimation: CBC Waveform Recovery\n"
        f"Match={match:.4f} | PSNR={psnr:.1f} dB | Corr={corr_complex:.4f} | "
        f"Mean RelErr={mean_rel_error:.1f}% | SNR={network_snr:.1f} | "
        f"Runtime={t_elapsed:.0f}s",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig_path = os.path.join(outdir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")
    
    try:
        result.plot_corner(
            parameters=["chirp_mass", "mass_ratio", "luminosity_distance"],
            filename=os.path.join(outdir, "corner_plot.png"),
            save=True,
        )
        print("Corner plot saved.")
    except Exception as e:
        print(f"Corner plot skipped: {e}")
    
    print("\n=== DONE ===")
    print(f"All outputs saved to {outdir}/")
    
    return metrics
