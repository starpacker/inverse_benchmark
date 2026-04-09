import sys
import os
import dill
import numpy as np
import traceback
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import irfft

# Import the target function
from agent_run_inversion import run_inversion


# ============================================================
# Injected Referee: evaluate_results (verbatim from Reference B)
# ============================================================
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


# ============================================================
# Main test logic
# ============================================================
def main():
    data_paths = ['/data/yjh/bilby_gw_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Number of args: {len(args)}, kwargs keys: {list(kwargs.keys())}")

    # Determine if chained execution
    is_chained = len(inner_paths) > 0

    if is_chained:
        # Pattern 2: Chained execution
        print("Detected CHAINED execution pattern.")
        print("Running run_inversion to get operator...")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)

            try:
                agent_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR running inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Pattern 1: Direct execution
        print("Detected DIRECT execution pattern.")
        print("Running run_inversion...")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        std_result = std_output

    # ============================================================
    # Evaluation Phase
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)

    # Extract data_dict and injection_parameters from the original args
    # run_inversion(data_dict, injection_parameters, outdir)
    if len(args) >= 2:
        data_dict = args[0]
        injection_parameters = args[1]
    else:
        data_dict = kwargs.get('data_dict', args[0] if len(args) > 0 else None)
        injection_parameters = kwargs.get('injection_parameters', args[1] if len(args) > 1 else None)

    if len(args) >= 3:
        outdir = args[2]
    else:
        outdir = kwargs.get('outdir', './test_output')

    # Create output directories for agent and standard evaluations
    outdir_agent = os.path.join(outdir, 'agent_eval') if outdir else './test_output/agent_eval'
    outdir_std = os.path.join(outdir, 'std_eval') if outdir else './test_output/std_eval'
    os.makedirs(outdir_agent, exist_ok=True)
    os.makedirs(outdir_std, exist_ok=True)

    # Infer duration, sampling_freq, minimum_freq from data_dict
    # These are standard bilby GW parameters
    freq_array = data_dict.get('freq_array', None)
    if freq_array is not None:
        df = freq_array[1] - freq_array[0]
        duration = 1.0 / df
        sampling_freq = 2.0 * freq_array[-1]
    else:
        duration = 4.0
        sampling_freq = 2048.0

    # Try to get minimum_freq from waveform_generator or default
    waveform_generator = data_dict.get('waveform_generator', None)
    minimum_freq = 20.0  # default
    if waveform_generator is not None:
        try:
            if hasattr(waveform_generator, 'frequency_domain_source_model'):
                pass
            if hasattr(waveform_generator, 'start_time'):
                pass
        except:
            pass

    print(f"Inferred parameters: duration={duration}, sampling_freq={sampling_freq}, minimum_freq={minimum_freq}")

    # Evaluate agent result
    print("\n--- Evaluating AGENT result ---")
    try:
        metrics_agent = evaluate_results(
            data_dict=data_dict,
            inversion_result=agent_result,
            injection_parameters=injection_parameters,
            duration=duration,
            sampling_freq=sampling_freq,
            minimum_freq=minimum_freq,
            outdir=outdir_agent,
        )
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard result
    print("\n--- Evaluating STANDARD result ---")
    try:
        metrics_std = evaluate_results(
            data_dict=data_dict,
            inversion_result=std_result,
            injection_parameters=injection_parameters,
            duration=duration,
            sampling_freq=sampling_freq,
            minimum_freq=minimum_freq,
            outdir=outdir_std,
        )
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ============================================================
    # Comparison and Verification
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Extract key metrics for comparison
    # Higher is better: psnr_db, correlation, waveform_match
    # Lower is better: mean_relative_error_pct

    psnr_agent = metrics_agent.get("psnr_db", 0.0)
    psnr_std = metrics_std.get("psnr_db", 0.0)

    corr_agent = metrics_agent.get("correlation", 0.0)
    corr_std = metrics_std.get("correlation", 0.0)

    match_agent = metrics_agent.get("waveform_match", 0.0)
    match_std = metrics_std.get("waveform_match", 0.0)

    err_agent = metrics_agent.get("mean_relative_error_pct", 100.0)
    err_std = metrics_std.get("mean_relative_error_pct", 100.0)

    print(f"PSNR (dB)          -> Agent: {psnr_agent:.2f}, Standard: {psnr_std:.2f}")
    print(f"Correlation        -> Agent: {corr_agent:.6f}, Standard: {corr_std:.6f}")
    print(f"Waveform Match     -> Agent: {match_agent:.6f}, Standard: {match_std:.6f}")
    print(f"Mean Rel Error (%) -> Agent: {err_agent:.2f}, Standard: {err_std:.2f}")

    # Determine pass/fail
    # We use multiple criteria with generous margins since this is stochastic (nested sampling)
    # For stochastic samplers, we allow significant margin
    passed = True
    reasons = []

    # PSNR: higher is better, allow 30% margin for stochastic sampling
    if psnr_std > 0 and psnr_agent < psnr_std * 0.7:
        reasons.append(f"PSNR too low: {psnr_agent:.2f} vs {psnr_std:.2f} (threshold: {psnr_std * 0.7:.2f})")
        passed = False

    # Correlation: higher is better, must be at least 0.9 of standard
    if corr_std > 0 and corr_agent < corr_std * 0.9:
        reasons.append(f"Correlation too low: {corr_agent:.6f} vs {corr_std:.6f}")
        passed = False

    # Waveform match: higher is better, must be at least 0.9 of standard
    if match_std > 0 and match_agent < match_std * 0.9:
        reasons.append(f"Waveform match too low: {match_agent:.6f} vs {match_std:.6f}")
        passed = False

    # Mean relative error: lower is better, allow up to 2x the standard error
    # But also check absolute threshold - if both are small, it's fine
    if err_std > 0 and err_agent > err_std * 2.0 and err_agent > 20.0:
        reasons.append(f"Mean relative error too high: {err_agent:.2f}% vs {err_std:.2f}%")
        passed = False

    # Additional absolute thresholds - the algorithm should produce reasonable results
    if match_agent < 0.8:
        reasons.append(f"Waveform match below absolute threshold 0.8: {match_agent:.6f}")
        passed = False

    if corr_agent < 0.8:
        reasons.append(f"Correlation below absolute threshold 0.8: {corr_agent:.6f}")
        passed = False

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASS - Agent performance is acceptable.")
        print(f"Scores -> Agent PSNR: {psnr_agent:.2f}, Standard PSNR: {psnr_std:.2f}")
        print(f"Scores -> Agent Match: {match_agent:.6f}, Standard Match: {match_std:.6f}")
        sys.exit(0)
    else:
        print("RESULT: FAIL - Agent performance degraded significantly.")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()