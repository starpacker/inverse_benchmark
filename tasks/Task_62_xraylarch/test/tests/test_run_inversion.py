import sys
import os
import dill
import numpy as np
import traceback
import json

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

# Inject the referee (evaluate_results) from Reference B
def evaluate_results(k, chi_meas, chi_clean, chi_fit, gt_shells, fit_shells,
                     k_weight, results_dir):
    """
    Compute metrics, save outputs, and generate visualization.
    
    Parameters
    ----------
    k : ndarray
        Photoelectron wavenumber array [Å^-1].
    chi_meas : ndarray
        Measured (noisy) EXAFS oscillation.
    chi_clean : ndarray
        Clean (ground truth) EXAFS oscillation.
    chi_fit : ndarray
        Fitted EXAFS oscillation.
    gt_shells : list of dict
        Ground truth shell parameters.
    fit_shells : list of dict
        Fitted shell parameters.
    k_weight : int
        k-weighting exponent.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    # k-weighted χ
    kw_gt = chi_clean * k**k_weight
    kw_fit = chi_fit * k**k_weight
    
    # Correlation coefficient
    cc = float(np.corrcoef(kw_gt, kw_fit)[0, 1])
    
    # RMSE
    rmse = float(np.sqrt(np.mean((kw_gt - kw_fit)**2)))
    
    # Data range and MSE
    dr = kw_gt.max() - kw_gt.min()
    mse = np.mean((kw_gt - kw_fit)**2)
    
    # PSNR
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    
    # 1-D SSIM: tile to make 2D (7×N) so win_size=7 works
    tile_rows = 7
    a2d = np.tile(kw_gt, (tile_rows, 1))
    b2d = np.tile(kw_fit, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))
    
    # Relative error
    re = float(np.linalg.norm(kw_gt - kw_fit) / max(np.linalg.norm(kw_gt), 1e-12))
    
    # R-space (FT) comparison
    window = np.hanning(len(k))
    ft_gt = np.abs(np.fft.fft(kw_gt * window))[:len(k)//2]
    ft_fit = np.abs(np.fft.fft(kw_fit * window))[:len(k)//2]
    cc_ft = float(np.corrcoef(ft_gt, ft_fit)[0, 1])
    
    # Parameter recovery metrics
    param_metrics = {}
    for i, (gt_sh, fit_sh) in enumerate(zip(gt_shells, fit_shells)):
        for key in ["N", "R", "sigma2"]:
            g, f = gt_sh[key], fit_sh[key]
            param_metrics[f"gt_{gt_sh['label']}_{key}"] = float(g)
            param_metrics[f"fit_{gt_sh['label']}_{key}"] = float(f)
            param_metrics[f"err_{gt_sh['label']}_{key}"] = float(abs(g - f))
    
    metrics = {
        "PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse,
        "CC_FT": cc_ft, **param_metrics
    }
    
    # Print metrics
    for key, val in sorted(metrics.items()):
        print(f"  {key:30s} = {val}")
    
    # Save metrics to JSON
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), chi_fit)
    np.save(os.path.join(results_dir, "ground_truth.npy"), chi_clean)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) k²χ(k)
    axes[0, 0].plot(k, chi_clean * k**2, 'b-', lw=2, label='GT')
    axes[0, 0].plot(k, chi_meas * k**2, 'k.', ms=1, alpha=0.3, label='Noisy')
    axes[0, 0].plot(k, chi_fit * k**2, 'r--', lw=1.5, label='Fit')
    axes[0, 0].set_xlabel('k [Å⁻¹]')
    axes[0, 0].set_ylabel('k²χ(k)')
    axes[0, 0].set_title('(a) EXAFS')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) Fourier transform (R-space)
    r = np.fft.fftfreq(len(k), d=(k[1] - k[0]) / (2 * np.pi))[:len(k)//2]
    axes[0, 1].plot(r, ft_gt, 'b-', lw=2, label='GT')
    axes[0, 1].plot(r, ft_fit, 'r--', lw=1.5, label='Fit')
    axes[0, 1].set_xlabel('R [Å]')
    axes[0, 1].set_ylabel('|FT|')
    axes[0, 1].set_title('(b) Radial Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 5)
    
    # (c) Residual
    axes[1, 0].plot(k, (chi_clean - chi_fit) * k**2, 'g-', lw=1)
    axes[1, 0].axhline(0, color='k', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('k [Å⁻¹]')
    axes[1, 0].set_ylabel('Residual k²Δχ')
    axes[1, 0].set_title(f'(c) Residual  RMSE={metrics["RMSE"]:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # (d) Parameter bars
    labels, gt_v, fit_v = [], [], []
    for gs, fs in zip(gt_shells, fit_shells):
        for key in ["N", "R", "sigma2"]:
            labels.append(f"{gs['label']}_{key}")
            gt_v.append(gs[key])
            fit_v.append(fs[key])
    x = np.arange(len(labels))
    w = 0.35
    axes[1, 1].bar(x - w/2, gt_v, w, label='GT', color='steelblue')
    axes[1, 1].bar(x + w/2, fit_v, w, label='Fit', color='tomato')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=7, rotation=30)
    axes[1, 1].set_title('(d) Parameters')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"xraylarch — EXAFS Fitting\nPSNR={metrics['PSNR']:.1f} dB  |  "
                 f"SSIM={metrics['SSIM']:.4f}  |  CC={metrics['CC']:.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics


# Helper: forward_operator (needed for generating ground truth chi)
def feff_amplitude(k, Z):
    """Simplified backscattering amplitude |f(k)|."""
    if Z == 8:  # O
        return 0.5 * np.exp(-0.01 * k**2) * (1 + 0.1 * np.sin(k))
    elif Z == 26:  # Fe
        return 0.8 * np.exp(-0.005 * k**2) * (1 + 0.2 * np.sin(1.5 * k))
    else:
        return 0.6 * np.exp(-0.008 * k**2)

def feff_phase(k, Z):
    """Simplified total phase shift δ(k)."""
    if Z == 8:
        return -0.2 * k + 0.5 + 0.02 * k**2
    elif Z == 26:
        return -0.3 * k + 1.0 + 0.015 * k**2
    else:
        return -0.25 * k + 0.7

def mean_free_path(k):
    """Mean free path λ(k) in Å."""
    return 1.0 / (0.003 * k**2 + 0.01)

def forward_operator(shells, k, s02=0.9):
    """
    Compute EXAFS χ(k) from shell parameters.
    """
    chi = np.zeros_like(k)
    lam = mean_free_path(k)
    
    for sh in shells:
        N = sh["N"]
        R = sh["R"]
        sig2 = sh["sigma2"]
        dE0 = sh.get("dE0", 0)
        Z = sh["Z"]
        
        k_eff = k
        
        amp = feff_amplitude(k_eff, Z)
        phase = feff_phase(k_eff, Z)
        
        chi += (N * s02 * amp / (k * R**2) *
                np.sin(2 * k * R + phase + 2 * k * dE0 * 0.01) *
                np.exp(-2 * sig2 * k**2) *
                np.exp(-2 * R / lam))
    
    return chi


def main():
    # Data paths
    data_paths = ['/data/yjh/xraylarch_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_data_path}")
    print(f"[INFO] Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer data
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data['args']
        kwargs = outer_data['kwargs']
        std_output = outer_data['output']
        
        print(f"[INFO] Loaded outer data for function: {outer_data['func_name']}")
        print(f"[INFO] Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
        
        # Execute the agent's run_inversion
        print("[INFO] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_data_paths:
            # Chained execution
            print("[INFO] Chained execution detected, loading inner data...")
            with open(inner_data_paths[0], 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data['args']
            inner_kwargs = inner_data['kwargs']
            std_result = inner_data['output']
            
            # Execute the operator
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # Extract results
        # run_inversion returns (fit_shells, chi_fit)
        agent_fit_shells, agent_chi_fit = final_result
        std_fit_shells, std_chi_fit = std_result
        
        # Extract input parameters
        k = args[0]
        chi_meas = args[1]
        s02 = args[2]
        k_weight = args[3]
        
        # Generate ground truth shells (typical Fe oxide structure)
        # These are the "true" parameters we're trying to recover
        gt_shells = [
            {"N": 6, "R": 2.0, "sigma2": 0.006, "dE0": 0, "Z": 8, "label": "Fe-O"},
            {"N": 2, "R": 3.0, "sigma2": 0.008, "dE0": 0, "Z": 26, "label": "Fe-Fe"},
        ]
        
        # Generate clean chi from ground truth
        chi_clean = forward_operator(gt_shells, k, s02)
        
        # Create results directories
        agent_results_dir = "./results_agent"
        std_results_dir = "./results_std"
        
        # Evaluate agent results
        print("\n[INFO] Evaluating Agent Results:")
        print("=" * 50)
        agent_metrics = evaluate_results(
            k, chi_meas, chi_clean, agent_chi_fit, gt_shells, agent_fit_shells,
            k_weight, agent_results_dir
        )
        
        # Evaluate standard results
        print("\n[INFO] Evaluating Standard Results:")
        print("=" * 50)
        std_metrics = evaluate_results(
            k, chi_meas, chi_clean, std_chi_fit, gt_shells, std_fit_shells,
            k_weight, std_results_dir
        )
        
        # Compare key metrics
        print("\n" + "=" * 50)
        print("[INFO] COMPARISON SUMMARY")
        print("=" * 50)
        
        key_metrics = ["PSNR", "SSIM", "CC", "RE", "RMSE", "CC_FT"]
        
        for metric in key_metrics:
            agent_val = agent_metrics.get(metric, 0)
            std_val = std_metrics.get(metric, 0)
            print(f"  {metric:10s}: Agent={agent_val:.6f}, Std={std_val:.6f}")
        
        # Determine success based on PSNR and CC (higher is better)
        # For RMSE and RE (lower is better)
        score_agent_psnr = agent_metrics.get("PSNR", 0)
        score_std_psnr = std_metrics.get("PSNR", 0)
        
        score_agent_cc = agent_metrics.get("CC", 0)
        score_std_cc = std_metrics.get("CC", 0)
        
        score_agent_rmse = agent_metrics.get("RMSE", float('inf'))
        score_std_rmse = std_metrics.get("RMSE", float('inf'))
        
        print(f"\nScores -> Agent PSNR: {score_agent_psnr:.4f}, Standard PSNR: {score_std_psnr:.4f}")
        print(f"Scores -> Agent CC: {score_agent_cc:.4f}, Standard CC: {score_std_cc:.4f}")
        print(f"Scores -> Agent RMSE: {score_agent_rmse:.6f}, Standard RMSE: {score_std_rmse:.6f}")
        
        # Check if agent performance is acceptable
        # Allow 10% margin for PSNR and CC, 20% margin for RMSE
        psnr_ok = score_agent_psnr >= score_std_psnr * 0.9
        cc_ok = score_agent_cc >= score_std_cc * 0.9
        rmse_ok = score_agent_rmse <= score_std_rmse * 1.2
        
        print(f"\n[INFO] PSNR acceptable: {psnr_ok}")
        print(f"[INFO] CC acceptable: {cc_ok}")
        print(f"[INFO] RMSE acceptable: {rmse_ok}")
        
        if psnr_ok and cc_ok and rmse_ok:
            print("\n[SUCCESS] Agent performance is acceptable!")
            sys.exit(0)
        else:
            print("\n[FAILURE] Agent performance degraded significantly!")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()