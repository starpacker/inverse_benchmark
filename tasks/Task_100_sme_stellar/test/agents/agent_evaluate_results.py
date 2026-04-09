import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_100_sme_stellar"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(data_dict, result_dict, results_dir, assets_dir):
    """
    Evaluate results: compute metrics, save outputs, and generate visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data
    result_dict : dict
        Dictionary from run_inversion
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
        
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics
    """
    wavelength = data_dict["wavelength"]
    flux_gt = data_dict["flux_gt"]
    flux_obs = data_dict["flux_obs"]
    gt_params = data_dict["gt_params"]
    
    flux_fit = result_dict["flux_fit"]
    fit_params = result_dict["fit_params"]
    
    gt_teff, gt_logg, gt_feh, gt_abundances = gt_params
    fit_teff, fit_logg, fit_feh, fit_abundances = fit_params
    
    # Compute spectrum metrics
    mse = np.mean((flux_gt - flux_fit)**2)
    psnr = 10.0 * np.log10(flux_gt.max()**2 / mse) if mse > 0 else 100.0
    cc = float(np.corrcoef(flux_gt.flatten(), flux_fit.flatten())[0, 1])
    
    # Compute parameter relative errors
    param_errors = {}
    
    # T_eff relative error
    if abs(gt_teff) > 1e-6:
        param_errors["RE_T_eff"] = abs(fit_teff - gt_teff) / abs(gt_teff)
    else:
        param_errors["RE_T_eff"] = abs(fit_teff - gt_teff)
    
    # log_g relative error
    if abs(gt_logg) > 1e-6:
        param_errors["RE_log_g"] = abs(fit_logg - gt_logg) / abs(gt_logg)
    else:
        param_errors["RE_log_g"] = abs(fit_logg - gt_logg)
    
    # [Fe/H] - use absolute error since it can be zero
    param_errors["RE_[Fe/H]"] = abs(fit_feh - gt_feh)
    
    # Abundance absolute errors
    for elem in sorted(gt_abundances.keys()):
        param_errors[f"AE_{elem}"] = abs(fit_abundances[elem] - gt_abundances[elem])
    
    # Build metrics dictionary
    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RE_Teff": float(param_errors["RE_T_eff"]),
        "RE_logg": float(param_errors["RE_log_g"]),
    }
    for elem in sorted(gt_abundances.keys()):
        metrics[f"AE_{elem}"] = float(param_errors[f"AE_{elem}"])
    
    # Print results
    print(f"    Spectrum PSNR = {psnr:.2f} dB")
    print(f"    Spectrum CC   = {cc:.6f}")
    print(f"    T_eff: GT={gt_teff:.0f} K, Fit={fit_teff:.0f} K, RE={param_errors['RE_T_eff']:.4f}")
    print(f"    log_g: GT={gt_logg:.2f}, Fit={fit_logg:.2f}, RE={param_errors['RE_log_g']:.4f}")
    print(f"    [Fe/H]: GT={gt_feh:.2f}, Fit={fit_feh:.2f}")
    for elem in sorted(gt_abundances.keys()):
        print(f"    [{elem}/H]: GT={gt_abundances[elem]:.2f}, Fit={fit_abundances[elem]:.2f}, AE={param_errors[f'AE_{elem}']:.4f}")
    
    # Save outputs
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), flux_gt)
        np.save(os.path.join(d, "recon_output.npy"), flux_fit)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: full spectrum overlay
    ax = axes[0, 0]
    ax.plot(wavelength, flux_obs, 'gray', alpha=0.4, lw=0.5, label='Observed')
    ax.plot(wavelength, flux_gt, 'b-', lw=1.0, label='GT spectrum')
    ax.plot(wavelength, flux_fit, 'r--', lw=1.0, label='Fitted spectrum')
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title("Stellar Spectrum: GT vs Fitted")
    ax.legend(fontsize=8)
    
    # Panel 2: zoom on Na D lines
    ax = axes[0, 1]
    mask = (wavelength > 5870) & (wavelength < 5920)
    ax.plot(wavelength[mask], flux_obs[mask], 'gray', alpha=0.5, lw=0.8, label='Observed')
    ax.plot(wavelength[mask], flux_gt[mask], 'b-', lw=1.2, label='GT')
    ax.plot(wavelength[mask], flux_fit[mask], 'r--', lw=1.2, label='Fitted')
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title("Zoom: Na D doublet (5890/5896 Å)")
    ax.legend(fontsize=8)
    
    # Panel 3: residuals
    ax = axes[1, 0]
    residual = flux_gt - flux_fit
    ax.plot(wavelength, residual, 'k-', lw=0.5)
    ax.axhline(0, color='r', ls='--', lw=0.5)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Residual (GT - Fit)")
    ax.set_title(f"Residuals | PSNR={psnr:.1f} dB, CC={cc:.4f}")
    
    # Panel 4: parameter comparison bar chart
    ax = axes[1, 1]
    labels = ["T_eff/1000", "log_g", "[Fe/H]+1"]
    gt_v = [gt_teff/1000, gt_logg, gt_feh+1]
    fit_v = [fit_teff/1000, fit_logg, fit_feh+1]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, gt_v, 0.3, label='GT', color='steelblue')
    ax.bar(x + 0.15, fit_v, 0.3, label='Fitted', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Stellar Parameters")
    ax.legend()
    
    plt.tight_layout()
    for path in [os.path.join(results_dir, "vis_result.png"),
                 os.path.join(assets_dir, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return metrics
