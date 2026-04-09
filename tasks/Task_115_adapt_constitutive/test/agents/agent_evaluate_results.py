import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_115_adapt_constitutive"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(
    strain,
    stress_gt,
    stress_noisy,
    stress_recon,
    params_true,
    params_fit
):
    """
    Compute metrics and generate visualizations.
    
    Metrics computed:
        - PSNR: Peak Signal-to-Noise Ratio on stress curve
        - CC: Correlation coefficient
        - RMSE: Root Mean Square Error
        - Parameter relative errors for E, sigma_y, K, n
    
    Args:
        strain: 1D array of strain values
        stress_gt: 1D array of ground-truth stress
        stress_noisy: 1D array of noisy stress observations
        stress_recon: 1D array of reconstructed stress
        params_true: true parameters [E, sigma_y, K, n]
        params_fit: fitted parameters [E, sigma_y, K, n]
    
    Returns:
        metrics: dictionary containing all computed metrics
    """
    # Compute metrics
    mse = np.mean((stress_gt - stress_recon) ** 2)
    data_range = np.max(stress_gt) - np.min(stress_gt)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))
    
    # Correlation coefficient
    cc = float(np.corrcoef(stress_gt.ravel(), stress_recon.ravel())[0, 1])
    
    # RMSE
    rmse = float(np.sqrt(mse))
    
    # Parameter relative errors
    names = ["E", "sigma_y", "K", "n"]
    param_errors = {}
    for name, pt, pf in zip(names, params_true, params_fit):
        re = abs(pf - pt) / abs(pt) * 100.0
        param_errors[f"{name}_true"] = float(pt)
        param_errors[f"{name}_fitted"] = float(pf)
        param_errors[f"{name}_RE_pct"] = float(re)
    
    metrics = {"PSNR": float(psnr), "CC": float(cc), "RMSE": float(rmse)}
    metrics.update(param_errors)
    
    # Print metrics
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.6f}")
    print(f"  RMSE = {metrics['RMSE']:.4f} MPa")
    for k in ["E", "sigma_y", "K", "n"]:
        print(f"  {k}: true={metrics[f'{k}_true']:.4f}  fit={metrics[f'{k}_fitted']:.4f}  RE={metrics[f'{k}_RE_pct']:.2f}%")
    
    # Save results
    print("[3/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), stress_gt)
        np.save(os.path.join(d, "recon_output.npy"), stress_recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    print("[4/4] Generating visualisation ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (a) GT vs Fitted stress-strain curve
    ax = axes[0]
    ax.plot(strain * 100, stress_gt, "k-", lw=2, label="Ground Truth")
    ax.plot(strain * 100, stress_noisy, ".", color="gray", ms=1, alpha=0.4, label="Noisy Observation")
    ax.plot(strain * 100, stress_recon, "r--", lw=2, label="Fitted Model")
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Stress-Strain Curve  (PSNR={metrics['PSNR']:.1f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Parameter comparison
    ax = axes[1]
    param_names = ["E (MPa)", "σ_y (MPa)", "K (MPa)", "n"]
    keys = ["E", "sigma_y", "K", "n"]
    true_vals = [metrics[f"{k}_true"] for k in keys]
    fit_vals = [metrics[f"{k}_fitted"] for k in keys]
    # Normalise for bar chart
    norm_true = [1.0] * 4
    norm_fit = [f / t if t != 0 else 0 for f, t in zip(fit_vals, true_vals)]
    x = np.arange(4)
    ax.bar(x - 0.18, norm_true, 0.35, label="True", color="steelblue")
    ax.bar(x + 0.18, norm_fit, 0.35, label="Fitted", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, fontsize=9)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Parameter Comparison (normalised)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # (c) Residual
    ax = axes[2]
    residual = stress_gt - stress_recon
    ax.plot(strain * 100, residual, "b-", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Residual Stress (MPa)")
    ax.set_title(f"Residual  (RMSE={metrics['RMSE']:.2f} MPa, CC={metrics['CC']:.4f})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    for p in [
        os.path.join(RESULTS_DIR, "reconstruction_result.png"),
        os.path.join(ASSETS_DIR, "reconstruction_result.png"),
        os.path.join(ASSETS_DIR, "vis_result.png")
    ]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics
