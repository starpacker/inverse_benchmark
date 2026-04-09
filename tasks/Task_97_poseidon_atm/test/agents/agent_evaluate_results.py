import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

R_JUP = 7.1492e7

def evaluate_results(data_dict, result_dict, save_plots=True):
    """
    Compute spectrum-fit and parameter-recovery metrics, and generate visualizations.

    Metrics:
      - PSNR: Peak signal-to-noise ratio of spectrum fit
      - CC: Pearson correlation coefficient
      - RMSE: Root mean square error of spectrum
      - RE: Relative error (norm)
      - Parameter-level relative errors

    Parameters
    ----------
    data_dict : dict
        Dictionary containing wavelengths, observed spectrum, clean spectrum,
        and configuration parameters.
    result_dict : dict
        Dictionary containing fit_params and spectrum_fit.
    save_plots : bool
        Whether to save visualization plots.

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    wavelengths = data_dict["wavelengths"]
    spectrum_obs = data_dict["spectrum_obs"]
    spectrum_clean = data_dict["spectrum_clean"]
    gt_params = data_dict["gt_params"]
    results_dir = data_dict["results_dir"]

    fit_params = result_dict["fit_params"]
    spectrum_fit = result_dict["spectrum_fit"]

    residual = spectrum_clean - spectrum_fit
    mse = np.mean(residual ** 2)
    rmse = float(np.sqrt(mse))

    # CC
    cc = float(np.corrcoef(spectrum_clean, spectrum_fit)[0, 1])

    # PSNR
    data_range = spectrum_clean.max() - spectrum_clean.min()
    psnr = float(10.0 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # SSIM (tile 1D to 2D for skimage)
    tile_rows = 7
    a2d = np.tile(spectrum_clean, (tile_rows, 1))
    b2d = np.tile(spectrum_fit, (tile_rows, 1))
    ssim = float(ssim_fn(a2d, b2d, data_range=data_range, win_size=7))

    # Relative error
    norm_gt = np.linalg.norm(spectrum_clean)
    re = float(np.linalg.norm(residual) / max(norm_gt, 1e-12))

    # Parameter recovery metrics
    param_keys = ["T", "log_X_H2O", "log_X_CH4", "log_X_CO2"]
    param_metrics = {}
    for k in param_keys:
        gt_v = gt_params[k]
        fit_v = fit_params[k]
        param_metrics[f"gt_{k}"] = float(gt_v)
        param_metrics[f"fit_{k}"] = float(fit_v)
        param_metrics[f"abs_err_{k}"] = float(abs(gt_v - fit_v))
        if abs(gt_v) > 1e-12:
            param_metrics[f"rel_err_{k}_pct"] = float(
                abs(gt_v - fit_v) / abs(gt_v) * 100
            )

    # Also compare R_p
    gt_rp = gt_params["R_p"] / R_JUP
    fit_rp = fit_params["R_p"] / R_JUP
    param_metrics["gt_R_p_Rjup"] = float(gt_rp)
    param_metrics["fit_R_p_Rjup"] = float(fit_rp)
    param_metrics["rel_err_R_p_pct"] = float(abs(gt_rp - fit_rp) / gt_rp * 100)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim,
        "CC": cc,
        "RMSE": rmse,
        "RE": re,
        **param_metrics,
    }

    # Print metrics
    print("\n[EVAL] Computing metrics ...")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.6g}")
        else:
            print(f"  {k:30s} = {v}")

    # Save outputs
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(results_dir, "gt_output.npy"), spectrum_clean)
    np.save(os.path.join(results_dir, "recon_output.npy"), spectrum_fit)
    np.save(os.path.join(results_dir, "measurements.npy"), spectrum_obs)

    print(f"\n[SAVE] gt_output.npy      → {results_dir}")
    print(f"[SAVE] recon_output.npy   → {results_dir}")
    print(f"[SAVE] measurements.npy   → {results_dir}")

    if save_plots:
        # Generate visualization
        wav_um = wavelengths * 1e6  # convert to μm
        depth_ppm_obs = spectrum_obs * 1e6
        depth_ppm_clean = spectrum_clean * 1e6
        depth_ppm_fit = spectrum_fit * 1e6

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # (a) Transmission spectrum
        ax = axes[0, 0]
        ax.scatter(wav_um, depth_ppm_obs, s=3, c='gray', alpha=0.4, label='Noisy obs')
        ax.plot(wav_um, depth_ppm_clean, 'b-', lw=1.5, label='Ground truth')
        ax.plot(wav_um, depth_ppm_fit, 'r--', lw=1.5, label='Retrieved')
        ax.set_xlabel('Wavelength [μm]')
        ax.set_ylabel('Transit Depth [ppm]')
        ax.set_title('(a) Transmission Spectrum')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (b) Residuals
        ax = axes[0, 1]
        res_ppm = (depth_ppm_clean - depth_ppm_fit)
        ax.plot(wav_um, res_ppm, 'g-', lw=0.8)
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('Wavelength [μm]')
        ax.set_ylabel('Residual [ppm]')
        ax.set_title(f'(b) Residuals  RMSE = {metrics["RMSE"]*1e6:.2f} ppm')
        ax.grid(True, alpha=0.3)

        # (c) Absorption feature zoom (1.0–3.0 μm)
        ax = axes[1, 0]
        mask = (wav_um >= 1.0) & (wav_um <= 3.5)
        ax.plot(wav_um[mask], depth_ppm_clean[mask], 'b-', lw=2, label='GT')
        ax.plot(wav_um[mask], depth_ppm_fit[mask], 'r--', lw=2, label='Retrieved')

        # Annotate absorption bands
        band_labels = [
            (1.4, 'H₂O'), (1.65, 'CH₄'), (1.85, 'H₂O'),
            (2.3, 'CH₄'), (2.7, 'H₂O'),
        ]
        ymin, ymax = ax.get_ylim()
        for bwav, blabel in band_labels:
            if 1.0 <= bwav <= 3.5:
                ax.axvline(bwav, color='purple', alpha=0.3, ls=':')
                ax.text(bwav, ymax - 0.05 * (ymax - ymin), blabel,
                        ha='center', va='top', fontsize=7, color='purple')
        ax.set_xlabel('Wavelength [μm]')
        ax.set_ylabel('Transit Depth [ppm]')
        ax.set_title('(c) Absorption Features (1–3.5 μm)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (d) Retrieved vs GT parameters (bar chart)
        ax = axes[1, 1]
        keys = ["T", "log_X_H2O", "log_X_CH4", "log_X_CO2", "R_p/R_Jup"]
        gt_vals = [
            gt_params["T"],
            gt_params["log_X_H2O"],
            gt_params["log_X_CH4"],
            gt_params["log_X_CO2"],
            gt_params["R_p"] / R_JUP,
        ]
        fit_vals = [
            fit_params["T"],
            fit_params["log_X_H2O"],
            fit_params["log_X_CH4"],
            fit_params["log_X_CO2"],
            fit_params["R_p"] / R_JUP,
        ]
        x = np.arange(len(keys))
        w = 0.35
        ax.bar(x - w / 2, gt_vals, w, label='GT', color='steelblue')
        ax.bar(x + w / 2, fit_vals, w, label='Retrieved', color='tomato')
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=8, rotation=15)
        ax.set_title('(d) Parameter Recovery')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            f"POSEIDON-inspired — Exoplanet Atmospheric Retrieval\n"
            f"PSNR = {metrics['PSNR']:.1f} dB  |  "
            f"SSIM = {metrics['SSIM']:.4f}  |  "
            f"CC = {metrics['CC']:.6f}  |  "
            f"RE = {metrics['RE']:.2e}",
            fontsize=13, fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_path = os.path.join(results_dir, "reconstruction_result.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIS] Saved → {save_path}")

    return metrics
