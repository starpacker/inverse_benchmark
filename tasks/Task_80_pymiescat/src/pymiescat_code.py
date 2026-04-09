"""
PyMieScatt - Mie Scattering Particle Size & Refractive Index Inversion
========================================================================
Task: Recover complex refractive index (n + ik) and particle diameter from
      Mie scattering efficiency spectra across multiple wavelengths
Repo: https://github.com/bsumlin/PyMieScatt
Paper: Sumlin et al., "Retrieving the aerosol complex refractive index
       using PyMieScatt" (JQSRT, 2018)

Inverse Problem:
    Forward: Given particle parameters (diameter d, refractive index m = n + ik),
             compute scattering/absorption efficiency spectra Qsca(λ), Qabs(λ)
             using Lorenz-Mie theory
    Inverse: From observed multi-wavelength Qsca(λ) and Qabs(λ) spectra,
             recover the complex refractive index m and diameter d

Usage:
    /data/yjh/pymiescat_env/bin/python pymiescat_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import minimize, differential_evolution

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# Wavelength range (nm) - UV to near-IR
WAVELENGTHS = np.linspace(300, 1000, 50)

# Ground truth particle parameters
GT_DIAMETER = 300.0       # nm
GT_N_REAL = 1.55          # Real part of refractive index
GT_K_IMAG = 0.02          # Imaginary part of refractive index
NOISE_LEVEL = 0.02        # 2% relative noise on measurements

# Inversion bounds
N_BOUNDS = (1.2, 2.0)     # Bounds for real refractive index
K_BOUNDS = (0.001, 0.5)   # Bounds for imaginary part
D_BOUNDS = (100, 600)     # Bounds for diameter (nm)


# ═══════════════════════════════════════════════════════════
# 2. Forward Operator (Mie Theory)
# ═══════════════════════════════════════════════════════════
def forward_operator(n_real, k_imag, diameter, wavelengths):
    """
    Compute Mie scattering/absorption efficiency spectra.
    
    Forward model: y = A(x) where
        x = (n, k, d) - refractive index and diameter
        y = (Qsca(λ), Qabs(λ)) - efficiency spectra
    
    Uses Lorenz-Mie theory via PyMieScatt.MieQ
    
    Args:
        n_real: Real part of refractive index
        k_imag: Imaginary part of refractive index  
        diameter: Particle diameter (nm)
        wavelengths: Array of wavelengths (nm)
    
    Returns:
        qsca: Scattering efficiency spectrum
        qabs: Absorption efficiency spectrum
        qext: Extinction efficiency spectrum
        g: Asymmetry parameter spectrum
    """
    import PyMieScatt as ps
    
    m = complex(n_real, k_imag)
    qsca = np.zeros(len(wavelengths))
    qabs = np.zeros(len(wavelengths))
    qext = np.zeros(len(wavelengths))
    g_param = np.zeros(len(wavelengths))
    
    for i, wl in enumerate(wavelengths):
        result = ps.MieQ(m, wl, diameter)
        # MieQ returns: (Qext, Qsca, Qabs, g, Qpr, Qback, Qratio)
        qext[i] = float(result[0])
        qsca[i] = float(result[1])
        qabs[i] = float(result[2])
        g_param[i] = float(result[3])
    
    return qsca, qabs, qext, g_param


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def generate_data():
    """
    Generate synthetic Mie scattering observations from known parameters.
    
    Returns:
        observations: dict with noisy Qsca, Qabs spectra
        ground_truth: dict with true parameters
    """
    print(f"  [FORWARD] Computing Mie spectra for d={GT_DIAMETER} nm, "
          f"m={GT_N_REAL}+{GT_K_IMAG}j")
    
    # Forward model: compute clean spectra
    qsca_clean, qabs_clean, qext_clean, g_clean = forward_operator(
        GT_N_REAL, GT_K_IMAG, GT_DIAMETER, WAVELENGTHS
    )
    
    # Add relative noise
    qsca_noisy = qsca_clean * (1 + NOISE_LEVEL * np.random.randn(len(WAVELENGTHS)))
    qabs_noisy = qabs_clean * (1 + NOISE_LEVEL * np.random.randn(len(WAVELENGTHS)))
    qext_noisy = qsca_noisy + qabs_noisy  # Extinction = scattering + absorption
    
    observations = {
        'wavelengths': WAVELENGTHS,
        'qsca': qsca_noisy,
        'qabs': qabs_noisy,
        'qext': qext_noisy,
    }
    
    ground_truth = {
        'n_real': GT_N_REAL,
        'k_imag': GT_K_IMAG,
        'diameter': GT_DIAMETER,
        'qsca_clean': qsca_clean,
        'qabs_clean': qabs_clean,
        'qext_clean': qext_clean,
        'g_clean': g_clean,
    }
    
    print(f"  [FORWARD] Qsca range: [{qsca_clean.min():.4f}, {qsca_clean.max():.4f}]")
    print(f"  [FORWARD] Qabs range: [{qabs_clean.min():.4f}, {qabs_clean.max():.4f}]")
    print(f"  [FORWARD] Qext range: [{qext_clean.min():.4f}, {qext_clean.max():.4f}]")
    
    return observations, ground_truth


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver (Least-Squares Optimization)
# ═══════════════════════════════════════════════════════════
def cost_function(params, wavelengths, qsca_obs, qabs_obs):
    """
    Least-squares cost function for Mie inversion.
    
    J(n, k, d) = Σ_λ [ (Qsca_obs(λ) - Qsca_model(λ))² / Qsca_obs(λ)²
                      + (Qabs_obs(λ) - Qabs_model(λ))² / Qabs_obs(λ)² ]
    """
    n_real, k_imag, diameter = params
    
    try:
        qsca_model, qabs_model, _, _ = forward_operator(
            n_real, k_imag, diameter, wavelengths
        )
        
        # Relative residuals (weighted least squares)
        qsca_weight = np.maximum(np.abs(qsca_obs), 1e-10)
        qabs_weight = np.maximum(np.abs(qabs_obs), 1e-10)
        
        res_sca = ((qsca_obs - qsca_model) / qsca_weight) ** 2
        res_abs = ((qabs_obs - qabs_model) / qabs_weight) ** 2
        
        return np.sum(res_sca) + np.sum(res_abs)
    except Exception:
        return 1e10


def reconstruct(observations):
    """
    Inverse problem: recover (n, k, d) from observed Mie spectra.
    
    Uses a two-stage approach:
    1. Global search via differential evolution
    2. Local refinement via L-BFGS-B
    
    Returns:
        recon_params: dict with recovered n, k, d
        recon_spectra: dict with reconstructed Qsca, Qabs spectra
    """
    wl = observations['wavelengths']
    qsca_obs = observations['qsca']
    qabs_obs = observations['qabs']
    
    bounds = [N_BOUNDS, K_BOUNDS, D_BOUNDS]
    
    # Stage 1: Global optimization (differential evolution)
    print("  [INV] Stage 1: Differential evolution global search...")
    result_global = differential_evolution(
        cost_function,
        bounds=bounds,
        args=(wl, qsca_obs, qabs_obs),
        seed=42,
        maxiter=200,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )
    print(f"  [INV] Global result: n={result_global.x[0]:.4f}, "
          f"k={result_global.x[1]:.6f}, d={result_global.x[2]:.2f} nm, "
          f"cost={result_global.fun:.6e}")
    
    # Stage 2: Local refinement
    print("  [INV] Stage 2: L-BFGS-B local refinement...")
    result_local = minimize(
        cost_function,
        x0=result_global.x,
        args=(wl, qsca_obs, qabs_obs),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12},
    )
    
    n_rec, k_rec, d_rec = result_local.x
    print(f"  [INV] Final result: n={n_rec:.4f}, k={k_rec:.6f}, "
          f"d={d_rec:.2f} nm, cost={result_local.fun:.6e}")
    
    # Compute reconstructed spectra
    qsca_rec, qabs_rec, qext_rec, g_rec = forward_operator(
        n_rec, k_rec, d_rec, wl
    )
    
    recon_params = {
        'n_real': float(n_rec),
        'k_imag': float(k_rec),
        'diameter': float(d_rec),
    }
    
    recon_spectra = {
        'qsca': qsca_rec,
        'qabs': qabs_rec,
        'qext': qext_rec,
        'g': g_rec,
    }
    
    return recon_params, recon_spectra


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(ground_truth, recon_params, recon_spectra, observations):
    """Compute comprehensive evaluation metrics."""
    # Parameter recovery errors
    n_error = abs(recon_params['n_real'] - ground_truth['n_real'])
    k_error = abs(recon_params['k_imag'] - ground_truth['k_imag'])
    d_error = abs(recon_params['diameter'] - ground_truth['diameter'])
    
    n_re = n_error / ground_truth['n_real']
    k_re = k_error / ground_truth['k_imag']
    d_re = d_error / ground_truth['diameter']
    
    # Spectral reconstruction quality
    gt_qsca = ground_truth['qsca_clean']
    gt_qabs = ground_truth['qabs_clean']
    gt_qext = ground_truth['qext_clean']
    
    rec_qsca = recon_spectra['qsca']
    rec_qabs = recon_spectra['qabs']
    rec_qext = recon_spectra['qext']
    
    # RMSE for spectra
    rmse_qsca = np.sqrt(np.mean((gt_qsca - rec_qsca)**2))
    rmse_qabs = np.sqrt(np.mean((gt_qabs - rec_qabs)**2))
    rmse_qext = np.sqrt(np.mean((gt_qext - rec_qext)**2))
    
    # Correlation coefficients
    cc_qsca = np.corrcoef(gt_qsca, rec_qsca)[0, 1]
    cc_qabs = np.corrcoef(gt_qabs, rec_qabs)[0, 1]
    cc_qext = np.corrcoef(gt_qext, rec_qext)[0, 1]
    
    # PSNR for Qext spectrum
    data_range = gt_qext.max() - gt_qext.min()
    mse_qext = np.mean((gt_qext - rec_qext)**2)
    psnr = 10 * np.log10(data_range**2 / mse_qext) if mse_qext > 0 else float('inf')
    
    # Overall relative error
    gt_all = np.concatenate([gt_qsca, gt_qabs])
    rec_all = np.concatenate([rec_qsca, rec_qabs])
    overall_re = np.sqrt(np.mean((gt_all - rec_all)**2)) / np.sqrt(np.mean(gt_all**2))
    
    return {
        'n_real_gt': ground_truth['n_real'],
        'n_real_recon': recon_params['n_real'],
        'n_real_error': float(n_error),
        'n_real_relative_error': float(n_re),
        'k_imag_gt': ground_truth['k_imag'],
        'k_imag_recon': recon_params['k_imag'],
        'k_imag_error': float(k_error),
        'k_imag_relative_error': float(k_re),
        'diameter_gt': ground_truth['diameter'],
        'diameter_recon': recon_params['diameter'],
        'diameter_error': float(d_error),
        'diameter_relative_error': float(d_re),
        'rmse_qsca': float(rmse_qsca),
        'rmse_qabs': float(rmse_qabs),
        'rmse_qext': float(rmse_qext),
        'cc_qsca': float(cc_qsca),
        'cc_qabs': float(cc_qabs),
        'cc_qext': float(cc_qext),
        'psnr': float(psnr),
        'rmse': float(rmse_qext),
        'overall_relative_error': float(overall_re),
    }


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(observations, ground_truth, recon_spectra, recon_params, metrics, save_path):
    """Generate comprehensive Mie scattering inversion visualization."""
    wl = observations['wavelengths']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Qsca spectrum comparison
    axes[0, 0].plot(wl, ground_truth['qsca_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 0].plot(wl, observations['qsca'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 0].plot(wl, recon_spectra['qsca'], 'r--', lw=2, label='Reconstructed')
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Qsca')
    axes[0, 0].set_title(f'Scattering Efficiency (CC={metrics["cc_qsca"]:.6f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) Qabs spectrum comparison
    axes[0, 1].plot(wl, ground_truth['qabs_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 1].plot(wl, observations['qabs'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 1].plot(wl, recon_spectra['qabs'], 'r--', lw=2, label='Reconstructed')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Qabs')
    axes[0, 1].set_title(f'Absorption Efficiency (CC={metrics["cc_qabs"]:.6f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) Qext spectrum comparison
    axes[0, 2].plot(wl, ground_truth['qext_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 2].plot(wl, observations['qext'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 2].plot(wl, recon_spectra['qext'], 'r--', lw=2, label='Reconstructed')
    axes[0, 2].set_xlabel('Wavelength (nm)')
    axes[0, 2].set_ylabel('Qext')
    axes[0, 2].set_title(f'Extinction Efficiency (CC={metrics["cc_qext"]:.6f})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # (d) Residuals
    res_sca = ground_truth['qsca_clean'] - recon_spectra['qsca']
    res_abs = ground_truth['qabs_clean'] - recon_spectra['qabs']
    axes[1, 0].plot(wl, res_sca, 'b-', lw=1.5, label='Qsca residual')
    axes[1, 0].plot(wl, res_abs, 'r-', lw=1.5, label='Qabs residual')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('Spectral Residuals (GT - Recon)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (e) Parameter comparison bar chart
    params_names = ['n (real)', 'k (imag)', 'd (nm)']
    gt_vals = [ground_truth['n_real'], ground_truth['k_imag'], ground_truth['diameter']]
    rec_vals = [recon_params['n_real'], recon_params['k_imag'], recon_params['diameter']]
    
    # Normalize for display
    gt_norm = np.array(gt_vals) / np.array(gt_vals)
    rec_norm = np.array(rec_vals) / np.array(gt_vals)
    
    x_pos = np.arange(len(params_names))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, gt_norm, width, label='Ground Truth', color='steelblue')
    axes[1, 1].bar(x_pos + width/2, rec_norm, width, label='Reconstructed', color='coral')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(params_names)
    axes[1, 1].set_ylabel('Normalized Value (GT=1.0)')
    axes[1, 1].set_title('Parameter Recovery')
    axes[1, 1].legend()
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add error annotations
    for i, (name, gt, rec) in enumerate(zip(params_names, gt_vals, rec_vals)):
        re = abs(rec - gt) / gt * 100
        axes[1, 1].annotate(f'{re:.2f}%', xy=(i, max(gt_norm[i], rec_norm[i]) + 0.02),
                            ha='center', fontsize=9, color='red')
    
    # (f) Asymmetry parameter
    axes[1, 2].plot(wl, ground_truth['g_clean'], 'b-', lw=2, label='GT g(λ)')
    axes[1, 2].plot(wl, recon_spectra['g'], 'r--', lw=2, label='Recon g(λ)')
    axes[1, 2].set_xlabel('Wavelength (nm)')
    axes[1, 2].set_ylabel('g (asymmetry parameter)')
    axes[1, 2].set_title('Asymmetry Parameter')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"PyMieScatt — Mie Scattering Refractive Index Inversion\n"
        f"GT: n={ground_truth['n_real']}, k={ground_truth['k_imag']}, d={ground_truth['diameter']} nm | "
        f"Recon: n={recon_params['n_real']:.4f}, k={recon_params['k_imag']:.6f}, d={recon_params['diameter']:.2f} nm\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC_ext={metrics['cc_qext']:.6f} | RE={metrics['overall_relative_error']:.6f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  PyMieScatt — Mie Scattering Inversion")
    print("=" * 60)
    
    # (a) Generate synthetic data
    print("\n[DATA] Generating synthetic Mie scattering observations...")
    observations, ground_truth = generate_data()
    print(f"[DATA] Wavelengths: {len(WAVELENGTHS)} points, "
          f"range [{WAVELENGTHS[0]:.0f}, {WAVELENGTHS[-1]:.0f}] nm")
    
    # (b) Run inversion
    print("\n[RECON] Running refractive index inversion...")
    recon_params, recon_spectra = reconstruct(observations)
    print(f"[RECON] Recovered: n={recon_params['n_real']:.6f} "
          f"(GT: {GT_N_REAL}), k={recon_params['k_imag']:.6f} "
          f"(GT: {GT_K_IMAG}), d={recon_params['diameter']:.2f} "
          f"(GT: {GT_DIAMETER})")
    
    # (c) Evaluate
    print("\n[EVAL] Computing evaluation metrics...")
    metrics = compute_metrics(ground_truth, recon_params, recon_spectra, observations)
    
    print(f"[EVAL] n error = {metrics['n_real_error']:.6f} "
          f"(RE: {metrics['n_real_relative_error']:.6f})")
    print(f"[EVAL] k error = {metrics['k_imag_error']:.6f} "
          f"(RE: {metrics['k_imag_relative_error']:.6f})")
    print(f"[EVAL] d error = {metrics['diameter_error']:.2f} nm "
          f"(RE: {metrics['diameter_relative_error']:.6f})")
    print(f"[EVAL] CC_Qsca = {metrics['cc_qsca']:.6f}")
    print(f"[EVAL] CC_Qabs = {metrics['cc_qabs']:.6f}")
    print(f"[EVAL] CC_Qext = {metrics['cc_qext']:.6f}")
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] Overall RE = {metrics['overall_relative_error']:.6f}")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Save arrays
    # Ground truth: clean spectra
    gt_spectra = np.stack([ground_truth['qsca_clean'], 
                           ground_truth['qabs_clean'],
                           ground_truth['qext_clean']], axis=0)
    # Reconstruction: recovered spectra
    rec_spectra = np.stack([recon_spectra['qsca'],
                            recon_spectra['qabs'],
                            recon_spectra['qext']], axis=0)
    # Input: noisy observations
    input_data = np.stack([observations['qsca'],
                           observations['qabs'],
                           observations['qext']], axis=0)
    
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_spectra)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), rec_spectra)
    np.save(os.path.join(RESULTS_DIR, "input.npy"), input_data)
    print(f"[SAVE] GT spectra shape: {gt_spectra.shape} → ground_truth.npy")
    print(f"[SAVE] Recon spectra shape: {rec_spectra.shape} → reconstruction.npy")
    print(f"[SAVE] Input spectra shape: {input_data.shape} → input.npy")
    
    # (f) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(observations, ground_truth, recon_spectra, recon_params, 
                      metrics, vis_path)
    
    print("\n" + "=" * 60)
    print("  DONE — PyMieScatt Mie Scattering Inversion")
    print("=" * 60)
