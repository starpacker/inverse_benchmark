import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_104_bisip_sip"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(freq, results):
    """
    Evaluate inversion results by computing metrics, generating plots,
    and saving outputs.
    
    Parameters:
        freq: ndarray, frequency array
        results: list of dicts, inversion results for each spectrum
    
    Returns:
        metrics: dict, average metrics (PSNR, CC)
    """
    
    def compute_spectrum_metrics(rho_true, rho_fit):
        """Compute PSNR and CC for spectrum comparison."""
        amp_true = np.abs(rho_true)
        amp_fit = np.abs(rho_fit)
        
        # Normalize
        amp_true_n = amp_true / amp_true.max()
        amp_fit_n = amp_fit / amp_fit.max()
        
        # PSNR
        mse = np.mean((amp_true_n - amp_fit_n) ** 2)
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
        
        # CC (correlation coefficient)
        t_z = amp_true_n - amp_true_n.mean()
        f_z = amp_fit_n - amp_fit_n.mean()
        denom = np.sqrt(np.sum(t_z ** 2) * np.sum(f_z ** 2))
        cc = np.sum(t_z * f_z) / denom if denom > 1e-15 else 0.0
        
        return float(psnr), float(cc)
    
    def compute_param_errors(gt_params, fit_params):
        """Compute relative errors for each Cole-Cole parameter."""
        errors = {}
        for key in ["rho0", "m", "tau", "c"]:
            rel_err = abs(fit_params[key] - gt_params[key]) / abs(gt_params[key]) * 100.0
            errors[key] = float(rel_err)
        return errors
    
    all_psnr = []
    all_cc = []
    
    for res in results:
        psnr, cc = compute_spectrum_metrics(res["rho_true"], res["rho_fit"])
        param_errors = compute_param_errors(res["gt"], res["fit"])
        
        res["psnr"] = psnr
        res["cc"] = cc
        res["param_errors"] = param_errors
        
        all_psnr.append(psnr)
        all_cc.append(cc)
        
        print(f"    PSNR={psnr:.1f} dB, CC={cc:.4f}")
        print(f"    Param errors: {param_errors}")
    
    # Average metrics
    avg_psnr = float(np.mean(all_psnr))
    avg_cc = float(np.mean(all_cc))
    print(f"\n[Summary] Avg PSNR = {avg_psnr:.2f} dB, Avg CC = {avg_cc:.4f}")
    
    metrics = {
        "PSNR": avg_psnr,
        "CC": avg_cc,
        "SSIM": "N/A (1D spectra)",
    }
    
    # Build gt_output and recon_output arrays
    gt_spectra = np.array([np.abs(r["rho_true"]) for r in results])
    recon_spectra = np.array([np.abs(r["rho_fit"]) for r in results])
    
    # Save outputs
    print("[4] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_spectra)
        np.save(os.path.join(d, "recon_output.npy"), recon_spectra)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate plots
    print("[5] Plotting ...")
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for i, res in enumerate(results):
        rho_true = res["rho_true"]
        rho_obs = res["rho_obs"]
        rho_fit = res["rho_fit"]
        
        amp_true = np.abs(rho_true)
        amp_obs = np.abs(rho_obs)
        amp_fit = np.abs(rho_fit)
        phase_true = -np.angle(rho_true) * 1000.0  # mrad
        phase_obs = -np.angle(rho_obs) * 1000.0
        phase_fit = -np.angle(rho_fit) * 1000.0
        
        # Amplitude plot
        ax = axes[i, 0]
        ax.semilogx(freq, amp_true, 'k-', lw=2, label='True')
        ax.semilogx(freq, amp_obs, 'b.', ms=4, alpha=0.5, label='Observed')
        ax.semilogx(freq, amp_fit, 'r--', lw=1.5, label='Fit')
        ax.set_ylabel("|ρ*| (Ω·m)")
        ax.set_title(f"Spectrum {i + 1}: ρ₀={res['gt']['rho0']:.0f}, "
                     f"m={res['gt']['m']:.1f}, τ={res['gt']['tau']:.2f}, "
                     f"c={res['gt']['c']:.1f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == n - 1:
            ax.set_xlabel("Frequency (Hz)")
        
        # Phase plot
        ax = axes[i, 1]
        ax.semilogx(freq, phase_true, 'k-', lw=2, label='True')
        ax.semilogx(freq, phase_obs, 'b.', ms=4, alpha=0.5, label='Observed')
        ax.semilogx(freq, phase_fit, 'r--', lw=1.5, label='Fit')
        ax.set_ylabel("-φ (mrad)")
        errs = res["param_errors"]
        ax.set_title(f"Errors: ρ₀={errs['rho0']:.1f}%, m={errs['m']:.1f}%, "
                     f"τ={errs['tau']:.1f}%, c={errs['c']:.1f}%", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == n - 1:
            ax.set_xlabel("Frequency (Hz)")
    
    plt.suptitle(f"Cole-Cole SIP Inversion — "
                 f"Avg PSNR={metrics['PSNR']:.1f}dB, CC={metrics['CC']:.3f}",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics
