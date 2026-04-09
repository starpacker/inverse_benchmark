import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# Define the paths
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_104_bisip_sip"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Inject the evaluate_results function (Reference B)
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


def main():
    # Data paths provided
    data_paths = ['/data/yjh/bisip_sip_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze data paths
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"[INFO] Outer data path: {outer_data_path}")
    print(f"[INFO] Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer data
        print("[1] Loading outer data...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'unknown')
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Function name: {func_name}")
        print(f"[INFO] Args count: {len(args)}")
        print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
        
        # Extract freq from args (first argument)
        freq = args[0] if len(args) > 0 else kwargs.get('freq', None)
        
        print("[2] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_data_paths:
            print("[INFO] Chained execution detected - processing inner data...")
            # Load inner data
            inner_data_path = inner_data_paths[0]
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        print("[3] Evaluating agent results...")
        # Evaluate agent results
        agent_metrics = evaluate_results(freq, final_result)
        
        print("\n[3b] Evaluating standard results...")
        # Evaluate standard results
        # Need to recalculate since evaluate_results modifies the input
        std_metrics_psnr = []
        std_metrics_cc = []
        
        for res in std_result:
            amp_true = np.abs(res["rho_true"])
            amp_fit = np.abs(res["rho_fit"])
            
            amp_true_n = amp_true / amp_true.max()
            amp_fit_n = amp_fit / amp_fit.max()
            
            mse = np.mean((amp_true_n - amp_fit_n) ** 2)
            psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
            
            t_z = amp_true_n - amp_true_n.mean()
            f_z = amp_fit_n - amp_fit_n.mean()
            denom = np.sqrt(np.sum(t_z ** 2) * np.sum(f_z ** 2))
            cc = np.sum(t_z * f_z) / denom if denom > 1e-15 else 0.0
            
            std_metrics_psnr.append(psnr)
            std_metrics_cc.append(cc)
        
        std_avg_psnr = float(np.mean(std_metrics_psnr))
        std_avg_cc = float(np.mean(std_metrics_cc))
        
        std_metrics = {
            "PSNR": std_avg_psnr,
            "CC": std_avg_cc
        }
        
        print(f"\n{'='*60}")
        print(f"[RESULTS] Agent Metrics: PSNR={agent_metrics['PSNR']:.2f} dB, CC={agent_metrics['CC']:.4f}")
        print(f"[RESULTS] Standard Metrics: PSNR={std_metrics['PSNR']:.2f} dB, CC={std_metrics['CC']:.4f}")
        print(f"{'='*60}")
        
        # Verification: Higher PSNR and CC are better
        # Allow a margin of 10% degradation
        psnr_threshold = std_metrics['PSNR'] * 0.9
        cc_threshold = std_metrics['CC'] * 0.9
        
        psnr_ok = agent_metrics['PSNR'] >= psnr_threshold
        cc_ok = agent_metrics['CC'] >= cc_threshold
        
        print(f"\n[CHECK] PSNR: Agent={agent_metrics['PSNR']:.2f}, Threshold={psnr_threshold:.2f} -> {'PASS' if psnr_ok else 'FAIL'}")
        print(f"[CHECK] CC: Agent={agent_metrics['CC']:.4f}, Threshold={cc_threshold:.4f} -> {'PASS' if cc_ok else 'FAIL'}")
        
        if psnr_ok and cc_ok:
            print("\n[SUCCESS] Agent performance is acceptable.")
            sys.exit(0)
        else:
            print("\n[FAILURE] Agent performance degraded significantly.")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()