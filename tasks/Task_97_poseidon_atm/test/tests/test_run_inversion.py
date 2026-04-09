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
# Inject the evaluate_results function (Reference B) verbatim
# ============================================================
R_JUP = 7.1492e7

def evaluate_results(data_dict, result_dict, save_plots=True):
    """
    Compute spectrum-fit and parameter-recovery metrics, and generate visualizations.
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
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(results_dir, "gt_output.npy"), spectrum_clean)
    np.save(os.path.join(results_dir, "recon_output.npy"), spectrum_fit)
    np.save(os.path.join(results_dir, "measurements.npy"), spectrum_obs)

    print(f"\n[SAVE] gt_output.npy      -> {results_dir}")
    print(f"[SAVE] recon_output.npy   -> {results_dir}")
    print(f"[SAVE] measurements.npy   -> {results_dir}")

    if save_plots:
        wav_um = wavelengths * 1e6
        depth_ppm_obs = spectrum_obs * 1e6
        depth_ppm_clean = spectrum_clean * 1e6
        depth_ppm_fit = spectrum_fit * 1e6

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        ax = axes[0, 0]
        ax.scatter(wav_um, depth_ppm_obs, s=3, c='gray', alpha=0.4, label='Noisy obs')
        ax.plot(wav_um, depth_ppm_clean, 'b-', lw=1.5, label='Ground truth')
        ax.plot(wav_um, depth_ppm_fit, 'r--', lw=1.5, label='Retrieved')
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Transit Depth [ppm]')
        ax.set_title('(a) Transmission Spectrum')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        res_ppm = (depth_ppm_clean - depth_ppm_fit)
        ax.plot(wav_um, res_ppm, 'g-', lw=0.8)
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Residual [ppm]')
        ax.set_title(f'(b) Residuals  RMSE = {metrics["RMSE"]*1e6:.2f} ppm')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        mask = (wav_um >= 1.0) & (wav_um <= 3.5)
        ax.plot(wav_um[mask], depth_ppm_clean[mask], 'b-', lw=2, label='GT')
        ax.plot(wav_um[mask], depth_ppm_fit[mask], 'r--', lw=2, label='Retrieved')

        band_labels = [
            (1.4, 'H2O'), (1.65, 'CH4'), (1.85, 'H2O'),
            (2.3, 'CH4'), (2.7, 'H2O'),
        ]
        ymin, ymax = ax.get_ylim()
        for bwav, blabel in band_labels:
            if 1.0 <= bwav <= 3.5:
                ax.axvline(bwav, color='purple', alpha=0.3, ls=':')
                ax.text(bwav, ymax - 0.05 * (ymax - ymin), blabel,
                        ha='center', va='top', fontsize=7, color='purple')
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Transit Depth [ppm]')
        ax.set_title('(c) Absorption Features (1-3.5 um)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

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
            f"POSEIDON-inspired -- Exoplanet Atmospheric Retrieval\n"
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
        print(f"[VIS] Saved -> {save_path}")

    return metrics


# ============================================================
# Main Test Logic
# ============================================================
def main():
    data_paths = ['/data/yjh/poseidon_atm_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

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

    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"[INFO] Function: {func_name}")
    print(f"[INFO] Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")

    # Determine execution pattern
    if len(inner_paths) > 0:
        # Pattern 2: Chained execution
        print(f"[INFO] Pattern 2: Chained execution with {len(inner_paths)} inner file(s)")
        
        # Run outer to get operator
        print("[INFO] Running run_inversion (outer) ...")
        agent_operator = run_inversion(*args, **kwargs)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"[INFO] Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        # Run operator
        print("[INFO] Running operator (inner) ...")
        agent_result = agent_operator(*inner_args, **inner_kwargs)

    else:
        # Pattern 1: Direct execution
        print("[INFO] Pattern 1: Direct execution")
        print("[INFO] Running run_inversion ...")
        agent_result = run_inversion(*args, **kwargs)
        std_result = std_output

    # Now evaluate both agent and standard results
    # We need data_dict for evaluate_results - it's the first arg
    data_dict = args[0] if len(args) > 0 else kwargs.get('data_dict', None)

    if data_dict is None:
        print("[ERROR] Could not extract data_dict from inputs")
        sys.exit(1)

    # Ensure results_dir exists for both evaluations
    results_dir_agent = os.path.join(
        data_dict.get("results_dir", "/tmp/poseidon_results"), "agent"
    )
    results_dir_std = os.path.join(
        data_dict.get("results_dir", "/tmp/poseidon_results"), "standard"
    )

    # Evaluate agent result
    print("\n" + "=" * 60)
    print("EVALUATING AGENT RESULT")
    print("=" * 60)
    data_dict_agent = dict(data_dict)
    data_dict_agent["results_dir"] = results_dir_agent
    os.makedirs(results_dir_agent, exist_ok=True)

    try:
        metrics_agent = evaluate_results(data_dict_agent, agent_result, save_plots=True)
    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard result
    print("\n" + "=" * 60)
    print("EVALUATING STANDARD RESULT")
    print("=" * 60)
    data_dict_std = dict(data_dict)
    data_dict_std["results_dir"] = results_dir_std
    os.makedirs(results_dir_std, exist_ok=True)

    try:
        metrics_std = evaluate_results(data_dict_std, std_result, save_plots=True)
    except Exception as e:
        print(f"[ERROR] Standard evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Compare metrics
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    psnr_agent = metrics_agent["PSNR"]
    psnr_std = metrics_std["PSNR"]
    ssim_agent = metrics_agent["SSIM"]
    ssim_std = metrics_std["SSIM"]
    cc_agent = metrics_agent["CC"]
    cc_std = metrics_std["CC"]
    re_agent = metrics_agent["RE"]
    re_std = metrics_std["RE"]
    rmse_agent = metrics_agent["RMSE"]
    rmse_std = metrics_std["RMSE"]

    print(f"  PSNR  -> Agent: {psnr_agent:.4f}, Standard: {psnr_std:.4f}")
    print(f"  SSIM  -> Agent: {ssim_agent:.6f}, Standard: {ssim_std:.6f}")
    print(f"  CC    -> Agent: {cc_agent:.8f}, Standard: {cc_std:.8f}")
    print(f"  RE    -> Agent: {re_agent:.6e}, Standard: {re_std:.6e}")
    print(f"  RMSE  -> Agent: {rmse_agent:.6e}, Standard: {rmse_std:.6e}")

    # Determine pass/fail
    # Higher is better: PSNR, SSIM, CC
    # Lower is better: RE, RMSE
    # Allow 10% margin for higher-is-better, 10% margin for lower-is-better

    passed = True
    margin = 0.10  # 10% tolerance

    # PSNR check (higher is better)
    if psnr_agent < psnr_std * (1.0 - margin):
        print(f"[FAIL] PSNR degraded: {psnr_agent:.4f} < {psnr_std * (1.0 - margin):.4f}")
        passed = False
    else:
        print(f"[PASS] PSNR acceptable")

    # SSIM check (higher is better)
    if ssim_agent < ssim_std * (1.0 - margin):
        print(f"[FAIL] SSIM degraded: {ssim_agent:.6f} < {ssim_std * (1.0 - margin):.6f}")
        passed = False
    else:
        print(f"[PASS] SSIM acceptable")

    # CC check (higher is better)
    if cc_agent < cc_std * (1.0 - margin):
        print(f"[FAIL] CC degraded: {cc_agent:.8f} < {cc_std * (1.0 - margin):.8f}")
        passed = False
    else:
        print(f"[PASS] CC acceptable")

    # RE check (lower is better)
    if re_agent > re_std * (1.0 + margin):
        print(f"[FAIL] RE degraded: {re_agent:.6e} > {re_std * (1.0 + margin):.6e}")
        passed = False
    else:
        print(f"[PASS] RE acceptable")

    # RMSE check (lower is better)
    if rmse_agent > rmse_std * (1.0 + margin):
        print(f"[FAIL] RMSE degraded: {rmse_agent:.6e} > {rmse_std * (1.0 + margin):.6e}")
        passed = False
    else:
        print(f"[PASS] RMSE acceptable")

    print("\n" + "=" * 60)
    if passed:
        print("[RESULT] ALL CHECKS PASSED - Agent performance is acceptable")
        sys.exit(0)
    else:
        print("[RESULT] SOME CHECKS FAILED - Agent performance degraded")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)