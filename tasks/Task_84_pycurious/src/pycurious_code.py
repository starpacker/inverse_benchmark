"""
pycurious — Curie Point Depth Inversion from Magnetic Anomaly Spectra
=====================================================================
Task: Recover Curie depth (bottom of magnetic sources) from magnetic anomaly grid
Repo: https://github.com/brmather/pycurious
Paper: Mather & Delhaye, "PyCurious: A Python module for computing the Curie 
       depth from the spectral analysis of magnetic anomaly data" (JOSS, 2019)

Inverse Problem:
    Forward: Given Curie depth parameters (beta, zt, dz, C), compute
             radial power spectrum Φ(k) via Bouligand et al. (2009)
    Inverse: From observed magnetic anomaly grid, compute radial power
             spectrum, then fit (beta, zt, dz, C) via nonlinear optimization

Usage:
    /data/yjh/pycurious_env/bin/python pycurious_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# Grid parameters (in metres)
NX = 256          # Grid points in x
NY = 256          # Grid points in y
DX = 1000.0       # Grid spacing (m) = 1 km
XMIN = 0.0
XMAX = NX * DX
YMIN = 0.0
YMAX = NY * DX

# True Curie depth parameters — single set for the grid
# beta = fractal parameter, zt = top of magnetic layer (km),
# dz = thickness (km), C = field constant
TRUE_BETA = 3.0
TRUE_ZT = 1.0       # km
TRUE_DZ = 25.0       # km -> Curie depth = zt + dz = 26 km
TRUE_C = 5.0

# Centroids to fit (sample multiple windows to test consistency)
CENTROIDS = [
    (80000, 80000),
    (176000, 80000),
    (80000, 176000),
    (176000, 176000),
    (128000, 128000),
]

WINDOW_SIZE = 100000.0  # Window size for spectral analysis (m)
NOISE_LEVEL = 0.02      # Noise fraction


# ═══════════════════════════════════════════════════════════
# 2. Synthetic Magnetic Anomaly Generation
# ═══════════════════════════════════════════════════════════
def generate_synthetic_magnetic_grid():
    """
    Generate a synthetic magnetic anomaly grid whose radial power
    spectrum follows the Bouligand et al. (2009) model.
    """
    from pycurious import bouligand2009
    
    # Create wavenumber grids
    kx = np.fft.fftfreq(NX, d=DX) * 2 * np.pi  # rad/m
    ky = np.fft.fftfreq(NY, d=DX) * 2 * np.pi  # rad/m
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    kh_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Avoid k=0
    kh_grid[0, 0] = 1e-10
    
    # Convert wavenumber to rad/km for bouligand2009
    kh_km = kh_grid * 1000.0  # rad/km
    
    # Compute log power spectrum from true parameters
    log_phi = bouligand2009(kh_km, TRUE_BETA, TRUE_ZT, TRUE_DZ, TRUE_C)
    phi = np.exp(log_phi)
    
    # Create random phase grid
    phase = 2 * np.pi * np.random.rand(NY, NX)
    
    # Construct Fourier coefficients with correct power and random phase
    amplitude = np.sqrt(phi)
    fourier_coeff = amplitude * np.exp(1j * phase)
    
    # DC component real
    fourier_coeff[0, 0] = np.abs(fourier_coeff[0, 0])
    
    # Inverse FFT to get spatial domain
    grid_clean = np.real(np.fft.ifft2(fourier_coeff))
    
    # Do NOT normalize — preserve the power spectrum
    # Add noise
    noise = NOISE_LEVEL * np.std(grid_clean) * np.random.randn(NY, NX)
    grid_noisy = grid_clean + noise
    
    true_curie = TRUE_ZT + TRUE_DZ
    print(f"  Grid shape: {grid_noisy.shape}")
    print(f"  Anomaly range: [{grid_noisy.min():.1f}, {grid_noisy.max():.1f}] nT")
    print(f"  Grid extent: {XMIN/1e3:.0f}-{XMAX/1e3:.0f} km x {YMIN/1e3:.0f}-{YMAX/1e3:.0f} km")
    print(f"  True params: beta={TRUE_BETA:.2f}, zt={TRUE_ZT:.2f} km, dz={TRUE_DZ:.2f} km")
    print(f"  True Curie depth: {true_curie:.2f} km")
    
    return grid_noisy, grid_clean


# ═══════════════════════════════════════════════════════════
# 3. Forward Operator
# ═══════════════════════════════════════════════════════════
def forward_operator(kh, beta, zt, dz, C):
    """
    Forward: Compute the theoretical radial power spectrum from
    Curie depth parameters using Bouligand et al. (2009) model.
    """
    from pycurious import bouligand2009
    return bouligand2009(kh, beta, zt, dz, C)


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: Spectral Fitting with pycurious
# ═══════════════════════════════════════════════════════════
def reconstruct(grid_noisy):
    """
    Inverse: Fit Curie depth parameters from magnetic anomaly grid.
    
    Uses pycurious CurieOptimise class which:
    1. Extracts subgrid around each centroid
    2. Computes radial power spectrum
    3. Fits Bouligand model via scipy.optimize.minimize
    """
    from pycurious import CurieOptimise
    
    # Create CurieOptimise object
    grid_obj = CurieOptimise(grid_noisy, XMIN, XMAX, YMIN, YMAX)
    
    results = []
    for i, (xc, yc) in enumerate(CENTROIDS):
        print(f"\n  [FIT] Centroid {i+1}: ({xc/1e3:.0f}, {yc/1e3:.0f}) km")
        
        try:
            beta_fit, zt_fit, dz_fit, C_fit = grid_obj.optimise(
                WINDOW_SIZE, xc, yc,
                beta=3.0, zt=1.0, dz=20.0, C=5.0,
                taper=np.hanning
            )
            
            curie_depth = zt_fit + dz_fit
            
            result = {
                'xc': xc, 'yc': yc,
                'beta': float(beta_fit),
                'zt': float(zt_fit),
                'dz': float(dz_fit),
                'C': float(C_fit),
                'curie_depth': float(curie_depth),
            }
            
            print(f"  [FIT] Fitted: beta={beta_fit:.2f}, zt={zt_fit:.2f}, "
                  f"dz={dz_fit:.2f}, C={C_fit:.2f}")
            print(f"  [FIT] Curie depth: {curie_depth:.2f} km "
                  f"(true: {TRUE_ZT + TRUE_DZ:.2f} km)")
            
        except Exception as e:
            print(f"  [FIT] ERROR: {e}")
            result = {
                'xc': xc, 'yc': yc,
                'beta': np.nan, 'zt': np.nan,
                'dz': np.nan, 'C': np.nan,
                'curie_depth': np.nan,
            }
        
        results.append(result)
    
    return results, grid_obj


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(fitted_results, grid_obj):
    """
    Evaluate Curie depth inversion quality.
    All centroids share the same true parameters.
    """
    from pycurious import bouligand2009
    
    true_curie = TRUE_ZT + TRUE_DZ
    
    fitted_curie = []
    fitted_betas = []
    fitted_zts = []
    fitted_dzs = []
    
    for r in fitted_results:
        if not np.isnan(r['curie_depth']):
            fitted_curie.append(r['curie_depth'])
            fitted_betas.append(r['beta'])
            fitted_zts.append(r['zt'])
            fitted_dzs.append(r['dz'])
    
    fit_cd = np.array(fitted_curie)
    
    # Mean fitted values
    mean_cd = np.mean(fit_cd)
    std_cd = np.std(fit_cd)
    mean_beta = np.mean(fitted_betas)
    mean_zt = np.mean(fitted_zts)
    mean_dz = np.mean(fitted_dzs)
    
    # Curie depth error
    rmse_cd = np.sqrt(np.mean((fit_cd - true_curie)**2))
    rel_err_cd = np.abs(mean_cd - true_curie) / true_curie
    
    # Parameter relative errors (mean fitted vs true)
    rel_err_beta = np.abs(mean_beta - TRUE_BETA) / TRUE_BETA
    rel_err_zt = np.abs(mean_zt - TRUE_ZT) / TRUE_ZT
    rel_err_dz = np.abs(mean_dz - TRUE_DZ) / TRUE_DZ
    
    # Spectral fit quality at first centroid
    xc, yc = CENTROIDS[0]
    subgrid = grid_obj.subgrid(WINDOW_SIZE, xc, yc)
    k, Phi_obs, sigma_Phi = grid_obj.radial_spectrum(subgrid, taper=np.hanning)
    
    # Forward model at fitted params
    fit_r = fitted_results[0]
    Phi_fit = bouligand2009(k, fit_r['beta'], fit_r['zt'], fit_r['dz'], fit_r['C'])
    
    # True model spectrum
    Phi_true = bouligand2009(k, TRUE_BETA, TRUE_ZT, TRUE_DZ, TRUE_C)
    
    # Spectral CC
    valid = np.isfinite(Phi_obs) & np.isfinite(Phi_fit)
    cc_spec = np.corrcoef(Phi_obs[valid], Phi_fit[valid])[0, 1] if np.sum(valid) > 2 else float('nan')
    
    # PSNR of spectrum (fitted vs observed)
    mse_spec = np.mean((Phi_obs[valid] - Phi_fit[valid])**2)
    range_spec = Phi_obs[valid].max() - Phi_obs[valid].min()
    psnr_spec = 10 * np.log10(range_spec**2 / mse_spec) if mse_spec > 0 else float('inf')
    
    metrics = {
        'psnr_spectrum': float(psnr_spec),
        'cc_spectrum': float(cc_spec),
        'rmse_curie_depth_km': float(rmse_cd),
        'mean_curie_depth_km': float(mean_cd),
        'std_curie_depth_km': float(std_cd),
        'true_curie_depth_km': float(true_curie),
        'rel_error_curie_depth': float(rel_err_cd),
        'rel_error_beta': float(rel_err_beta),
        'rel_error_zt': float(rel_err_zt),
        'rel_error_dz': float(rel_err_dz),
        'mean_beta': float(mean_beta),
        'mean_zt': float(mean_zt),
        'mean_dz': float(mean_dz),
        'n_centroids_fitted': len(fit_cd),
    }
    
    return metrics


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(grid_noisy, grid_clean, fitted_results,
                      grid_obj, metrics, save_path):
    """Generate comprehensive visualization."""
    from pycurious import bouligand2009
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Magnetic anomaly grid
    ax = axes[0, 0]
    im = ax.imshow(grid_noisy, cmap='RdBu_r', aspect='equal', 
                    extent=[XMIN/1e3, XMAX/1e3, YMIN/1e3, YMAX/1e3], origin='lower')
    for xc, yc in CENTROIDS:
        ax.plot(xc/1e3, yc/1e3, 'k+', ms=15, mew=2)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Magnetic Anomaly Grid (nT)')
    plt.colorbar(im, ax=ax, label='nT')
    
    # (b) Radial power spectrum fit (centroid 1)
    ax = axes[0, 1]
    xc, yc = CENTROIDS[0]
    subgrid = grid_obj.subgrid(WINDOW_SIZE, xc, yc)
    k, Phi_obs, sigma_Phi = grid_obj.radial_spectrum(subgrid, taper=np.hanning)
    
    fit_r = fitted_results[0]
    Phi_fit = bouligand2009(k, fit_r['beta'], fit_r['zt'], fit_r['dz'], fit_r['C'])
    Phi_true = bouligand2009(k, TRUE_BETA, TRUE_ZT, TRUE_DZ, TRUE_C)
    
    ax.plot(k, Phi_obs, 'ko', ms=3, alpha=0.5, label='Observed')
    ax.plot(k, Phi_true, 'b-', lw=2, alpha=0.7, label='True model')
    ax.plot(k, Phi_fit, 'r--', lw=2, label='Fitted model')
    ax.set_xlabel('Wavenumber (rad/km)')
    ax.set_ylabel('ln(Power)')
    ax.set_title('Radial Power Spectrum (Centroid 1)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (c) Curie depth bar chart across centroids
    ax = axes[0, 2]
    true_cd = TRUE_ZT + TRUE_DZ
    fit_cds = [r['curie_depth'] for r in fitted_results if not np.isnan(r['curie_depth'])]
    x_pos = range(len(fit_cds))
    ax.bar(x_pos, fit_cds, color='steelblue', alpha=0.7, label='Fitted')
    ax.axhline(true_cd, color='r', ls='--', lw=2, label=f'True ({true_cd:.1f} km)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i+1}' for i in range(len(fit_cds))])
    ax.set_ylabel('Curie Depth (km)')
    ax.set_title('Curie Depth Recovery per Centroid')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # (d) Parameter recovery bar chart
    ax = axes[1, 0]
    param_names = ['β', 'z_t', 'Δz', 'z_Curie']
    rel_errors = [
        metrics['rel_error_beta'] * 100,
        metrics['rel_error_zt'] * 100,
        metrics['rel_error_dz'] * 100,
        metrics['rel_error_curie_depth'] * 100,
    ]
    bars = ax.bar(range(len(param_names)), rel_errors, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylabel('Mean Relative Error (%)')
    ax.set_title('Parameter Recovery Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rel_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # (e) Radial spectrum fit (centroid at center)
    ax = axes[1, 1]
    xc2, yc2 = CENTROIDS[-1]  # center centroid
    subgrid2 = grid_obj.subgrid(WINDOW_SIZE, xc2, yc2)
    k2, Phi_obs2, _ = grid_obj.radial_spectrum(subgrid2, taper=np.hanning)
    fit_r2 = fitted_results[-1]
    Phi_fit2 = bouligand2009(k2, fit_r2['beta'], fit_r2['zt'], fit_r2['dz'], fit_r2['C'])
    Phi_true2 = bouligand2009(k2, TRUE_BETA, TRUE_ZT, TRUE_DZ, TRUE_C)
    
    ax.plot(k2, Phi_obs2, 'ko', ms=3, alpha=0.5, label='Observed')
    ax.plot(k2, Phi_true2, 'b-', lw=2, alpha=0.7, label='True model')
    ax.plot(k2, Phi_fit2, 'r--', lw=2, label='Fitted model')
    ax.set_xlabel('Wavenumber (rad/km)')
    ax.set_ylabel('ln(Power)')
    ax.set_title('Radial Power Spectrum (Center)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (f) Parameter table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Parameter', 'True', 'Mean Fitted', 'Rel Err (%)'],
        ['β', f'{TRUE_BETA:.2f}', f'{metrics["mean_beta"]:.2f}', 
         f'{metrics["rel_error_beta"]*100:.1f}'],
        ['z_t (km)', f'{TRUE_ZT:.2f}', f'{metrics["mean_zt"]:.2f}',
         f'{metrics["rel_error_zt"]*100:.1f}'],
        ['Δz (km)', f'{TRUE_DZ:.2f}', f'{metrics["mean_dz"]:.2f}',
         f'{metrics["rel_error_dz"]*100:.1f}'],
        ['z_Curie (km)', f'{TRUE_ZT+TRUE_DZ:.2f}', f'{metrics["mean_curie_depth_km"]:.2f}',
         f'{metrics["rel_error_curie_depth"]*100:.1f}'],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Parameter Recovery Summary')
    
    fig.suptitle(
        f"pycurious — Curie Point Depth Inversion\n"
        f"PSNR_spec={metrics['psnr_spectrum']:.2f} dB | CC_spec={metrics['cc_spectrum']:.4f} | "
        f"RMSE_CD={metrics['rmse_curie_depth_km']:.2f} km | RE_CD={metrics['rel_error_curie_depth']*100:.1f}%",
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
    print("  pycurious — Curie Point Depth Inversion")
    print("=" * 60)
    
    # (a) Generate synthetic magnetic anomaly grid
    print("\n[DATA] Generating synthetic magnetic anomaly grid...")
    grid_noisy, grid_clean = generate_synthetic_magnetic_grid()
    
    # (b) Run Curie depth inversion
    print("\n[RECON] Running Curie depth inversion at multiple centroids...")
    fitted_results, grid_obj = reconstruct(grid_noisy)
    
    # (c) Evaluate
    print("\n\n[EVAL] Computing evaluation metrics...")
    metrics = compute_metrics(fitted_results, grid_obj)
    
    print(f"[EVAL] PSNR (spectrum) = {metrics['psnr_spectrum']:.4f} dB")
    print(f"[EVAL] CC (spectrum) = {metrics['cc_spectrum']:.6f}")
    print(f"[EVAL] RMSE (Curie depth) = {metrics['rmse_curie_depth_km']:.4f} km")
    print(f"[EVAL] Mean Curie depth = {metrics['mean_curie_depth_km']:.2f} km (true: {TRUE_ZT+TRUE_DZ:.2f} km)")
    print(f"[EVAL] Rel Error (Curie depth) = {metrics['rel_error_curie_depth']*100:.4f}%")
    print(f"[EVAL] Rel Error (beta) = {metrics['rel_error_beta']*100:.4f}%")
    print(f"[EVAL] Rel Error (zt) = {metrics['rel_error_zt']*100:.4f}%")
    print(f"[EVAL] Rel Error (dz) = {metrics['rel_error_dz']*100:.4f}%")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Save arrays
    np.save(os.path.join(RESULTS_DIR, "input.npy"), grid_noisy)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), grid_clean)
    
    # Save fitted Curie depths
    curie_data = np.array([[r['xc'], r['yc'], r['curie_depth'], r['beta'], r['zt'], r['dz']] 
                           for r in fitted_results])
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), curie_data)
    print(f"[SAVE] Input shape: {grid_noisy.shape} → input.npy")
    print(f"[SAVE] GT shape: {grid_clean.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {curie_data.shape} → reconstruction.npy")
    
    # (f) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(grid_noisy, grid_clean, fitted_results,
                      grid_obj, metrics, vis_path)
    
    print("\n" + "=" * 60)
    print("  DONE — pycurious Curie Point Depth Inversion")
    print("=" * 60)
