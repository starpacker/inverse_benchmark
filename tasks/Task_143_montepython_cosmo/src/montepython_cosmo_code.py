"""
montepython_cosmo - Cosmological Parameter Estimation via MCMC
==============================================================
Task: From CMB/BAO/SNe data, constrain cosmological parameters via Bayesian MCMC
Repo: https://github.com/brinckmann/montepython_public

Usage:
    /data/yjh/montepython_cosmo_env/bin/python montepython_cosmo_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.integrate import quad
from scipy.optimize import minimize

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. Cosmological Model: Flat ΛCDM
# ═══════════════════════════════════════════════════════════
# Constants
c_km_s = 299792.458  # speed of light in km/s

# True parameters (Planck 2018 best-fit)
TRUE_PARAMS = {
    'H0': 67.36,       # km/s/Mpc
    'Omega_m': 0.3153,  # total matter density
    'Omega_b': 0.0493,  # baryon density
}

def E_z(z, H0, Omega_m):
    """Dimensionless Hubble parameter E(z) = H(z)/H0 for flat ΛCDM."""
    Omega_Lambda = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def comoving_distance(z, H0, Omega_m):
    """Comoving distance in Mpc."""
    integrand = lambda zp: 1.0 / E_z(zp, H0, Omega_m)
    result, _ = quad(integrand, 0, z)
    return c_km_s / H0 * result

def luminosity_distance(z, H0, Omega_m):
    """Luminosity distance in Mpc."""
    return (1 + z) * comoving_distance(z, H0, Omega_m)

def angular_diameter_distance(z, H0, Omega_m):
    """Angular diameter distance in Mpc."""
    return comoving_distance(z, H0, Omega_m) / (1 + z)

def distance_modulus(z, H0, Omega_m):
    """Distance modulus μ = 5 log10(dL/10pc)."""
    dL = luminosity_distance(z, H0, Omega_m)  # Mpc
    return 5 * np.log10(dL * 1e6 / 10)  # convert Mpc to pc

def sound_horizon_approx(H0, Omega_m, Omega_b):
    """Approximate sound horizon at drag epoch (Mpc)."""
    h = H0 / 100.0
    omega_m = Omega_m * h**2
    omega_b = Omega_b * h**2
    # Eisenstein & Hu 1998 fitting formula (simplified)
    r_d = 147.09 * (omega_m / 0.1432)**(-0.255) * (omega_b / 0.02236)**(-0.128)
    return r_d

def D_V(z, H0, Omega_m):
    """Volume-averaged distance D_V(z) in Mpc."""
    d_C = comoving_distance(z, H0, Omega_m)
    return (d_C**2 * c_km_s * z / (H0 * E_z(z, H0, Omega_m)))**(1.0/3.0)

# ═══════════════════════════════════════════════════════════
# 2. Data Generation
# ═══════════════════════════════════════════════════════════
def generate_data():
    """Generate synthetic cosmological data with known parameters."""
    H0 = TRUE_PARAMS['H0']
    Om = TRUE_PARAMS['Omega_m']
    Ob = TRUE_PARAMS['Omega_b']
    
    np.random.seed(42)
    data = {}
    
    # --- CMB Distance Priors (Planck 2018) ---
    # Shift parameter R = sqrt(Omega_m) * d_C(z_star) * H0/c
    z_star = 1089.92  # recombination redshift
    d_C_star = comoving_distance(z_star, H0, Om)
    R_true = np.sqrt(Om) * H0 / c_km_s * d_C_star
    
    # Acoustic scale la = pi * d_A(z_star) / r_s
    r_s = sound_horizon_approx(H0, Om, Ob)
    d_A_star = d_C_star / (1 + z_star)
    la_true = np.pi * d_A_star / r_s
    
    omega_b_true = Ob * (H0/100)**2
    
    # Observed with Planck-like errors
    data['cmb'] = {
        'R': R_true + 0.0046 * np.random.randn(),
        'R_err': 0.0046,
        'la': la_true + 0.090 * np.random.randn(),
        'la_err': 0.090,
        'omega_b': omega_b_true + 0.00015 * np.random.randn(),
        'omega_b_err': 0.00015,
    }
    
    # --- BAO Data (BOSS DR12) ---
    bao_z = np.array([0.38, 0.51, 0.61])
    bao_DV_rd = np.array([
        D_V(z, H0, Om) / r_s for z in bao_z
    ])
    bao_err = np.array([0.12, 0.15, 0.17])  # fractional errors ~1-2%
    bao_obs = bao_DV_rd * (1 + bao_err * np.random.randn(len(bao_z)) * 0.01)
    
    data['bao'] = {
        'z': bao_z,
        'DV_rd_obs': bao_obs,
        'DV_rd_err': bao_DV_rd * 0.015,  # 1.5% errors
    }
    
    # --- Type Ia Supernovae (simplified Pantheon-like) ---
    sn_z = np.linspace(0.01, 1.5, 40)
    sn_mu_true = np.array([distance_modulus(z, H0, Om) for z in sn_z])
    sn_mu_err = 0.15 * np.ones(len(sn_z))  # typical SN scatter
    sn_mu_obs = sn_mu_true + sn_mu_err * np.random.randn(len(sn_z))
    
    data['sn'] = {
        'z': sn_z,
        'mu_obs': sn_mu_obs,
        'mu_err': sn_mu_err,
    }
    
    # Ground truth arrays
    gt_data = np.concatenate([
        [R_true, la_true, omega_b_true],  # CMB
        bao_DV_rd,  # BAO
        sn_mu_true,  # SN
    ])
    
    return data, gt_data

# ═══════════════════════════════════════════════════════════
# 3. Likelihood
# ═══════════════════════════════════════════════════════════
def log_likelihood(params, data):
    """Total log-likelihood from CMB + BAO + SN."""
    H0, Omega_m, Omega_b = params
    
    # Priors
    if H0 < 50 or H0 > 90:
        return -np.inf
    if Omega_m < 0.1 or Omega_m > 0.6:
        return -np.inf
    if Omega_b < 0.01 or Omega_b > 0.1:
        return -np.inf
    if Omega_b > Omega_m:
        return -np.inf
    
    try:
        logL = 0.0
        
        # --- CMB likelihood ---
        z_star = 1089.92
        d_C_star = comoving_distance(z_star, H0, Omega_m)
        R_model = np.sqrt(Omega_m) * H0 / c_km_s * d_C_star
        r_s = sound_horizon_approx(H0, Omega_m, Omega_b)
        d_A_star = d_C_star / (1 + z_star)
        la_model = np.pi * d_A_star / r_s
        omega_b_model = Omega_b * (H0/100)**2
        
        cmb = data['cmb']
        logL -= 0.5 * ((R_model - cmb['R']) / cmb['R_err'])**2
        logL -= 0.5 * ((la_model - cmb['la']) / cmb['la_err'])**2
        logL -= 0.5 * ((omega_b_model - cmb['omega_b']) / cmb['omega_b_err'])**2
        
        # --- BAO likelihood ---
        bao = data['bao']
        for i, z in enumerate(bao['z']):
            DV_model = D_V(z, H0, Omega_m)
            DV_rd_model = DV_model / r_s
            logL -= 0.5 * ((DV_rd_model - bao['DV_rd_obs'][i]) / bao['DV_rd_err'][i])**2
        
        # --- SN likelihood ---
        sn = data['sn']
        # Marginalize over absolute magnitude M (nuisance parameter)
        mu_model = np.array([distance_modulus(z, H0, Omega_m) for z in sn['z']])
        delta_mu = sn['mu_obs'] - mu_model
        weights = 1.0 / sn['mu_err']**2
        # Analytical marginalization over offset
        M_best = np.sum(delta_mu * weights) / np.sum(weights)
        chi2_sn = np.sum(((delta_mu - M_best) / sn['mu_err'])**2)
        logL -= 0.5 * chi2_sn
        
        return logL
        
    except Exception:
        return -np.inf

def log_prior(params):
    """Flat prior."""
    H0, Omega_m, Omega_b = params
    if 50 < H0 < 90 and 0.1 < Omega_m < 0.6 and 0.01 < Omega_b < 0.1 and Omega_b < Omega_m:
        return 0.0
    return -np.inf

def log_posterior(params, data):
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data)

# ═══════════════════════════════════════════════════════════
# 4. MCMC Sampling
# ═══════════════════════════════════════════════════════════
def run_mcmc(data, n_walkers=32, n_steps=4000, n_burn=1000):
    """Run MCMC using emcee."""
    import emcee
    
    ndim = 3  # H0, Omega_m, Omega_b
    
    # Initial positions: scatter around true values
    p0 = np.array([TRUE_PARAMS['H0'], TRUE_PARAMS['Omega_m'], TRUE_PARAMS['Omega_b']])
    pos = p0 + 0.1 * p0 * np.random.randn(n_walkers, ndim)
    # Ensure valid
    pos[:, 0] = np.clip(pos[:, 0], 55, 85)
    pos[:, 1] = np.clip(pos[:, 1], 0.15, 0.55)
    pos[:, 2] = np.clip(pos[:, 2], 0.02, 0.08)
    
    print(f"[MCMC] Starting: {n_walkers} walkers, {n_steps} steps")
    t0 = time.time()
    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args=(data,))
    sampler.run_mcmc(pos, n_steps, progress=False)
    
    elapsed = time.time() - t0
    print(f"[MCMC] Completed in {elapsed:.1f} s")
    
    # Discard burn-in and thin
    chain = sampler.get_chain(discard=n_burn, flat=True)
    
    # Best-fit: maximum posterior
    log_probs = sampler.get_log_prob(discard=n_burn, flat=True)
    best_idx = np.argmax(log_probs)
    best_params = chain[best_idx]
    
    # Posterior statistics
    param_names = ['H0', 'Omega_m', 'Omega_b']
    results = {}
    for i, name in enumerate(param_names):
        median = np.median(chain[:, i])
        std = np.std(chain[:, i])
        q16, q84 = np.percentile(chain[:, i], [16, 84])
        results[name] = {
            'median': float(median),
            'std': float(std),
            'best_fit': float(best_params[i]),
            'q16': float(q16),
            'q84': float(q84),
        }
    
    return chain, sampler.get_chain(discard=n_burn), results, elapsed

# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(results, data, gt_data):
    """Compute parameter errors and data fit quality."""
    H0_fit = results['H0']['median']
    Om_fit = results['Omega_m']['median']
    Ob_fit = results['Omega_b']['median']
    
    # Parameter relative errors
    param_errors = {}
    true_vals = [TRUE_PARAMS['H0'], TRUE_PARAMS['Omega_m'], TRUE_PARAMS['Omega_b']]
    fit_vals = [H0_fit, Om_fit, Ob_fit]
    names = ['H0', 'Omega_m', 'Omega_b']
    
    for name, true_val, fit_val in zip(names, true_vals, fit_vals):
        re = abs(true_val - fit_val) / abs(true_val)
        param_errors[name] = {
            'true': true_val,
            'estimated': round(fit_val, 6),
            'relative_error_pct': round(re * 100, 4),
            'std': round(results[name]['std'], 6),
        }
    
    mean_re_pct = np.mean([param_errors[n]['relative_error_pct'] for n in names])
    
    # Model fit PSNR
    # Compute model predictions at best-fit
    sn = data['sn']
    mu_model = np.array([distance_modulus(z, H0_fit, Om_fit) for z in sn['z']])
    mu_true = np.array([distance_modulus(z, TRUE_PARAMS['H0'], TRUE_PARAMS['Omega_m']) for z in sn['z']])
    
    data_range = mu_true.max() - mu_true.min()
    mse = np.mean((mu_true - mu_model)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    cc = float(np.corrcoef(mu_true, mu_model)[0, 1])
    
    recon_data = np.concatenate([
        [results['H0']['median'], results['Omega_m']['median'], results['Omega_b']['median']],
        mu_model,
    ])
    
    return {
        "psnr_dB": float(psnr),
        "correlation": float(cc),
        "mean_parameter_relative_error_pct": float(mean_re_pct),
        "parameter_estimates": param_errors,
        "method": "MCMC_emcee_simplified_cosmological_likelihood"
    }, recon_data

# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize(chain, chain_3d, results, data, metrics, save_path):
    """Create visualization with corner plot and diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    param_names = ['H₀', 'Ωₘ', 'Ωᵦ']
    true_vals = [TRUE_PARAMS['H0'], TRUE_PARAMS['Omega_m'], TRUE_PARAMS['Omega_b']]
    
    # (a) Chain convergence traces
    ax = axes[0, 0]
    n_steps = chain_3d.shape[0]
    for i, (name, true_val) in enumerate(zip(param_names, true_vals)):
        ax_sub = ax if i == 0 else ax.twinx() if i == 1 else ax
        colors = ['blue', 'red', 'green']
        for walker in range(min(5, chain_3d.shape[1])):
            ax.plot(chain_3d[:, walker, 0], alpha=0.3, color='blue', lw=0.5)
    ax.axhline(TRUE_PARAMS['H0'], color='black', ls='--', lw=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('H₀ (km/s/Mpc)')
    ax.set_title('(a) MCMC Chain Traces (H₀)')
    
    # (b) Posterior: H0 vs Omega_m
    ax = axes[0, 1]
    ax.scatter(chain[:, 0], chain[:, 1], c='blue', alpha=0.01, s=1)
    ax.axvline(TRUE_PARAMS['H0'], color='red', ls='--', lw=1.5, label='True')
    ax.axhline(TRUE_PARAMS['Omega_m'], color='red', ls='--', lw=1.5)
    ax.scatter([results['H0']['median']], [results['Omega_m']['median']], 
               c='red', s=100, marker='*', zorder=5, label='Median')
    ax.set_xlabel('H₀ (km/s/Mpc)')
    ax.set_ylabel('Ωₘ')
    ax.set_title('(b) Posterior: H₀ vs Ωₘ')
    ax.legend()
    
    # (c) SN Ia Hubble diagram fit
    ax = axes[1, 0]
    sn = data['sn']
    H0_fit = results['H0']['median']
    Om_fit = results['Omega_m']['median']
    mu_model = np.array([distance_modulus(z, H0_fit, Om_fit) for z in sn['z']])
    mu_true = np.array([distance_modulus(z, TRUE_PARAMS['H0'], TRUE_PARAMS['Omega_m']) for z in sn['z']])
    
    ax.errorbar(sn['z'], sn['mu_obs'], yerr=sn['mu_err'], fmt='k.', ms=3, alpha=0.5, label='Observed')
    ax.plot(sn['z'], mu_true, 'b-', lw=2, label='True model')
    ax.plot(sn['z'], mu_model, 'r--', lw=2, label='Best fit')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance modulus μ')
    ax.set_title('(c) SN Ia Hubble Diagram')
    ax.legend()
    
    # (d) Parameter comparison bar chart
    ax = axes[1, 1]
    names_short = ['H₀', 'Ωₘ', 'Ωᵦ']
    medians = [results[k]['median'] for k in ['H0', 'Omega_m', 'Omega_b']]
    stds = [results[k]['std'] for k in ['H0', 'Omega_m', 'Omega_b']]
    
    # Normalize for comparison
    true_arr = np.array(true_vals)
    fit_arr = np.array(medians)
    
    x_pos = np.arange(len(names_short))
    width = 0.35
    ax.bar(x_pos - width/2, true_arr / true_arr, width, label='True', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, fit_arr / true_arr, width, label='MCMC Median', color='red', alpha=0.7,
           yerr=np.array(stds)/true_arr, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_short, fontsize=12)
    ax.set_ylabel('Relative to True')
    ax.set_title('(d) Parameter Recovery')
    ax.legend()
    ax.axhline(1.0, color='k', ls=':', alpha=0.3)
    ax.set_ylim([0.95, 1.05])
    
    pe = metrics['parameter_estimates']
    title = (f"Cosmological MCMC | PSNR={metrics['psnr_dB']:.1f} dB | CC={metrics['correlation']:.6f}\n"
             f"H₀={pe['H0']['estimated']:.2f}±{pe['H0']['std']:.2f} | "
             f"Ωₘ={pe['Omega_m']['estimated']:.4f}±{pe['Omega_m']['std']:.4f} | "
             f"Ωᵦ={pe['Omega_b']['estimated']:.4f}±{pe['Omega_b']['std']:.4f}")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  montepython_cosmo — Cosmological MCMC Pipeline")
    print("=" * 60)
    
    # (a) Generate data
    data, gt_data = generate_data()
    print(f"[DATA] CMB: R={data['cmb']['R']:.4f}, la={data['cmb']['la']:.3f}")
    print(f"[DATA] BAO: {len(data['bao']['z'])} measurements")
    print(f"[DATA] SN: {len(data['sn']['z'])} supernovae")
    
    # (b) Run MCMC
    chain, chain_3d, results, elapsed = run_mcmc(data, n_walkers=32, n_steps=4000, n_burn=1000)
    print(f"[MCMC] {len(chain)} posterior samples")
    for name in ['H0', 'Omega_m', 'Omega_b']:
        r = results[name]
        print(f"[MCMC] {name}: {r['median']:.4f} ± {r['std']:.4f} (true: {TRUE_PARAMS[name]})")
    
    # (c) Metrics
    metrics, recon_data = compute_metrics(results, data, gt_data)
    print(f"[EVAL] PSNR = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC = {metrics['correlation']:.6f}")
    print(f"[EVAL] Mean parameter RE = {metrics['mean_parameter_relative_error_pct']:.4f}%")
    
    # (d) Save metrics
    metrics['runtime_seconds'] = elapsed
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")
    
    # (e) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize(chain, chain_3d, results, data, metrics, vis_path)
    
    # (f) Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), recon_data)
    
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
