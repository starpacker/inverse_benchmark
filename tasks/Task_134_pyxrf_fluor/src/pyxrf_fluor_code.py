"""
pyxrf_fluor - XRF Fluorescence Spectrum Deconvolution
=====================================================
Task: Deconvolve XRF fluorescence energy spectrum to quantify elemental composition
Repo: https://github.com/NSLS-II/PyXRF
Paper: Li et al., Proc. SPIE 2017, doi:10.1117/12.2272585

Inverse Problem:
    Given a measured XRF spectrum S(E), decompose it into individual elemental
    fluorescence line contributions to quantify elemental concentrations:
    S(E) = Σ_k c_k · L_k(E; E_k, σ_k) + B(E) + noise

Usage:
    /data/yjh/pyxrf_fluor_env/bin/python pyxrf_fluor_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.optimize import minimize, curve_fit
from lmfit import Model, Parameters

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# XRF Characteristic Lines (keV) — common elements
# Format: {element: [(line_name, energy_keV, relative_intensity), ...]}
XRF_LINES = {
    'Fe': [('Ka', 6.404, 1.0), ('Kb', 7.058, 0.17)],
    'Cu': [('Ka', 8.048, 1.0), ('Kb', 8.905, 0.17)],
    'Zn': [('Ka', 8.639, 1.0), ('Kb', 9.572, 0.17)],
    'Ca': [('Ka', 3.692, 1.0), ('Kb', 4.013, 0.13)],
    'Ti': [('Ka', 4.511, 1.0), ('Kb', 4.932, 0.14)],
    'Cr': [('Ka', 5.415, 1.0), ('Kb', 5.947, 0.15)],
    'Mn': [('Ka', 5.899, 1.0), ('Kb', 6.490, 0.16)],
    'Ni': [('Ka', 7.472, 1.0), ('Kb', 8.265, 0.17)],
}

# Energy grid
E_MIN = 1.0   # keV
E_MAX = 12.0  # keV
E_STEP = 0.01 # keV (10 eV resolution)
DETECTOR_FWHM = 0.15  # keV — detector energy resolution

# Elements to include in synthetic sample
SAMPLE_ELEMENTS = ['Fe', 'Cu', 'Zn', 'Ca', 'Ti', 'Mn']
# Ground truth concentrations (arbitrary units, proportional to peak area)
GT_CONCENTRATIONS = {
    'Fe': 100.0,
    'Cu': 60.0,
    'Zn': 45.0,
    'Ca': 80.0,
    'Ti': 35.0,
    'Mn': 50.0,
}

# ═══════════════════════════════════════════════════════════
# 2. Forward Model: XRF Spectrum Generation
# ═══════════════════════════════════════════════════════════
def gaussian_peak(energy, center, amplitude, sigma):
    """Gaussian peak profile for detector-broadened XRF line."""
    return amplitude * np.exp(-0.5 * ((energy - center) / sigma) ** 2)

def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def generate_element_spectrum(energy, element, concentration, det_sigma):
    """
    Generate XRF spectrum for a single element.
    Each characteristic line is a Gaussian broadened by detector resolution.
    """
    spectrum = np.zeros_like(energy)
    if element not in XRF_LINES:
        return spectrum
    
    for line_name, line_energy, rel_intensity in XRF_LINES[element]:
        amplitude = concentration * rel_intensity
        spectrum += gaussian_peak(energy, line_energy, amplitude, det_sigma)
    
    return spectrum

def generate_background(energy, a=500.0, b=2.5, c=0.1):
    """
    Generate Bremsstrahlung background: exponential + scatter.
    B(E) = a * exp(-b*E) + c
    """
    return a * np.exp(-b * energy) + c

def forward_operator(energy, concentrations, det_fwhm=DETECTOR_FWHM):
    """
    Forward model: concentrations → XRF spectrum
    S(E) = Σ_k c_k · Σ_lines G(E; E_line, σ_det) · I_rel + B(E)
    """
    det_sigma = fwhm_to_sigma(det_fwhm)
    spectrum = np.zeros_like(energy)
    
    for element, conc in concentrations.items():
        spectrum += generate_element_spectrum(energy, element, conc, det_sigma)
    
    return spectrum

# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """
    Generate synthetic XRF spectrum with known elemental concentrations.
    Returns: (noisy_spectrum, gt_concentrations_array, metadata)
    """
    energy = np.arange(E_MIN, E_MAX, E_STEP)
    
    # Generate clean element spectra
    clean_signal = forward_operator(energy, GT_CONCENTRATIONS)
    
    # Generate background
    background = generate_background(energy)
    
    # Total clean spectrum
    clean_spectrum = clean_signal + background
    
    # Add Poisson-like noise (photon counting statistics)
    # Scale to realistic count rates
    scale_factor = 10.0  # counts per unit
    expected_counts = clean_spectrum * scale_factor
    noisy_counts = np.random.poisson(np.maximum(expected_counts, 1).astype(int)).astype(float)
    noisy_spectrum = noisy_counts / scale_factor
    
    # GT as array
    gt_array = np.array([GT_CONCENTRATIONS[el] for el in SAMPLE_ELEMENTS])
    
    metadata = {
        'energy': energy,
        'clean_signal': clean_signal,
        'background': background,
        'clean_spectrum': clean_spectrum,
        'elements': SAMPLE_ELEMENTS,
        'gt_concentrations': GT_CONCENTRATIONS,
        'det_fwhm': DETECTOR_FWHM,
    }
    
    print(f"[DATA] Generated XRF spectrum with {len(SAMPLE_ELEMENTS)} elements")
    print(f"[DATA] Energy range: {energy[0]:.1f} - {energy[-1]:.2f} keV")
    print(f"[DATA] GT concentrations: {GT_CONCENTRATIONS}")
    
    return noisy_spectrum, gt_array, metadata

# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: XRF Spectrum Deconvolution
# ═══════════════════════════════════════════════════════════
def build_xrf_model(energy, elements, det_fwhm):
    """
    Build the multi-element XRF fitting model using lmfit.
    Model: Σ_k c_k · element_spectrum_k(E) + background(E)
    """
    det_sigma = fwhm_to_sigma(det_fwhm)
    
    # Pre-compute element basis spectra (unit concentration)
    basis_spectra = {}
    for element in elements:
        basis_spectra[element] = generate_element_spectrum(energy, element, 1.0, det_sigma)
    
    return basis_spectra

def reconstruct(noisy_spectrum, metadata):
    """
    XRF spectrum deconvolution: recover elemental concentrations.
    
    Algorithm:
    1. Estimate and subtract background
    2. Build element basis spectra (unit concentration)
    3. Non-negative least squares fitting: S = Σ c_k · B_k + background
    4. Refine with lmfit for uncertainties
    """
    energy = metadata['energy']
    elements = metadata['elements']
    det_fwhm = metadata['det_fwhm']
    det_sigma = fwhm_to_sigma(det_fwhm)
    
    # Step 1: Build element basis spectra
    basis_spectra = build_xrf_model(energy, elements, det_fwhm)
    
    # Step 2: Construct design matrix [B_1, B_2, ..., B_k, bg_exp, bg_const]
    n_elements = len(elements)
    n_energy = len(energy)
    
    # Design matrix: each column is an element's basis spectrum
    A = np.zeros((n_energy, n_elements + 2))
    for i, element in enumerate(elements):
        A[:, i] = basis_spectra[element]
    
    # Background basis functions
    A[:, n_elements] = np.exp(-2.5 * energy)  # exponential background
    A[:, n_elements + 1] = np.ones(n_energy)  # constant background
    
    # Step 3: Non-negative least squares
    from scipy.optimize import nnls
    coeffs, residual = nnls(A, noisy_spectrum)
    
    fitted_concentrations = {}
    for i, element in enumerate(elements):
        fitted_concentrations[element] = float(coeffs[i])
    
    bg_amp = coeffs[n_elements]
    bg_const = coeffs[n_elements + 1]
    
    print(f"[RECON] NNLS fitted concentrations: {fitted_concentrations}")
    
    # Step 4: Refine with lmfit for better fit and uncertainties
    params = Parameters()
    for i, element in enumerate(elements):
        params.add(f'c_{element}', value=coeffs[i], min=0)
    params.add('bg_amp', value=bg_amp * 500, min=0)
    params.add('bg_decay', value=2.5, min=0.5, max=5.0)
    params.add('bg_const', value=bg_const, min=0)
    
    def residual_func(params, energy, data, elements, basis_spectra):
        model = np.zeros_like(energy)
        for element in elements:
            model += params[f'c_{element}'].value * basis_spectra[element]
        model += params['bg_amp'].value * np.exp(-params['bg_decay'].value * energy)
        model += params['bg_const'].value
        return (data - model)
    
    from lmfit import minimize as lm_minimize
    result = lm_minimize(residual_func, params, args=(energy, noisy_spectrum, elements, basis_spectra))
    
    # Extract refined concentrations
    refined_concentrations = {}
    for element in elements:
        refined_concentrations[element] = float(result.params[f'c_{element}'].value)
    
    print(f"[RECON] Refined concentrations: {refined_concentrations}")
    
    # Build reconstructed spectrum
    recon_spectrum = np.zeros_like(energy)
    for element in elements:
        recon_spectrum += refined_concentrations[element] * basis_spectra[element]
    recon_spectrum += result.params['bg_amp'].value * np.exp(-result.params['bg_decay'].value * energy)
    recon_spectrum += result.params['bg_const'].value
    
    # Build fitted background
    fitted_bg = result.params['bg_amp'].value * np.exp(-result.params['bg_decay'].value * energy) + result.params['bg_const'].value
    
    return {
        'nnls_concentrations': fitted_concentrations,
        'refined_concentrations': refined_concentrations,
        'recon_spectrum': recon_spectrum,
        'fitted_background': fitted_bg,
        'basis_spectra': basis_spectra,
        'fit_result': result,
    }

# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_concentrations, recon_result, clean_spectrum, recon_spectrum):
    """Compute evaluation metrics for XRF deconvolution."""
    refined = recon_result['refined_concentrations']
    elements = list(gt_concentrations.keys())
    
    metrics = {}
    
    # Per-element concentration errors
    gt_array = []
    fitted_array = []
    rel_errors = []
    
    for el in elements:
        gt_val = gt_concentrations[el]
        fit_val = refined.get(el, 0.0)
        gt_array.append(gt_val)
        fitted_array.append(fit_val)
        re = abs(fit_val - gt_val) / gt_val * 100.0
        rel_errors.append(re)
        metrics[f'{el}_gt'] = gt_val
        metrics[f'{el}_fitted'] = round(fit_val, 4)
        metrics[f'{el}_RE_pct'] = round(re, 4)
    
    gt_array = np.array(gt_array)
    fitted_array = np.array(fitted_array)
    
    # Overall concentration metrics
    metrics['mean_RE_pct'] = float(np.mean(rel_errors))
    metrics['max_RE_pct'] = float(np.max(rel_errors))
    
    # Concentration CC
    cc = float(np.corrcoef(gt_array, fitted_array)[0, 1])
    metrics['concentration_CC'] = cc
    
    # Concentration R²
    ss_res = np.sum((gt_array - fitted_array) ** 2)
    ss_tot = np.sum((gt_array - np.mean(gt_array)) ** 2)
    metrics['concentration_R2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    # Concentration PSNR
    data_range = gt_array.max() - gt_array.min()
    mse_conc = np.mean((gt_array - fitted_array) ** 2)
    if mse_conc > 0:
        metrics['concentration_PSNR'] = float(10 * np.log10(data_range ** 2 / mse_conc))
    else:
        metrics['concentration_PSNR'] = float('inf')
    
    # Spectrum-level metrics
    data_range_spec = clean_spectrum.max() - clean_spectrum.min()
    mse_spec = np.mean((clean_spectrum - recon_spectrum) ** 2)
    if mse_spec > 0:
        metrics['spectrum_PSNR'] = float(10 * np.log10(data_range_spec ** 2 / mse_spec))
    else:
        metrics['spectrum_PSNR'] = float('inf')
    
    cc_spec = float(np.corrcoef(clean_spectrum, recon_spectrum)[0, 1])
    metrics['spectrum_CC'] = cc_spec
    
    rmse_spec = float(np.sqrt(mse_spec))
    metrics['spectrum_RMSE'] = rmse_spec
    
    return metrics

# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(energy, noisy_spectrum, clean_spectrum, recon_result, 
                      gt_concentrations, metrics, save_path):
    """Generate 4-panel visualization for XRF deconvolution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    recon_spectrum = recon_result['recon_spectrum']
    refined = recon_result['refined_concentrations']
    basis_spectra = recon_result['basis_spectra']
    fitted_bg = recon_result['fitted_background']
    elements = list(gt_concentrations.keys())
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))
    
    # Panel 1: Noisy spectrum with individual element contributions (GT)
    ax1 = axes[0, 0]
    ax1.plot(energy, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.5, label='Noisy data')
    ax1.plot(energy, clean_spectrum, 'b-', alpha=0.8, linewidth=1.0, label='Clean total')
    for i, el in enumerate(elements):
        det_sigma = fwhm_to_sigma(DETECTOR_FWHM)
        el_spec = generate_element_spectrum(energy, el, gt_concentrations[el], det_sigma)
        ax1.fill_between(energy, 0, el_spec, alpha=0.3, color=colors[i], label=f'{el} (GT)')
    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Intensity (counts)')
    ax1.set_title('(a) GT XRF Spectrum — Element Contributions')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_xlim(E_MIN, E_MAX)
    
    # Panel 2: Fitted spectrum decomposition
    ax2 = axes[0, 1]
    ax2.plot(energy, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.5, label='Noisy data')
    ax2.plot(energy, recon_spectrum, 'r-', alpha=0.8, linewidth=1.0, label='Fitted total')
    ax2.plot(energy, fitted_bg, 'g--', alpha=0.6, linewidth=1.0, label='Background')
    for i, el in enumerate(elements):
        el_spec = refined[el] * basis_spectra[el]
        ax2.fill_between(energy, 0, el_spec, alpha=0.3, color=colors[i], label=f'{el} (fit)')
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel('Intensity (counts)')
    psnr = metrics.get('spectrum_PSNR', 0)
    cc = metrics.get('spectrum_CC', 0)
    ax2.set_title(f'(b) Fitted XRF Decomposition — PSNR={psnr:.2f} dB, CC={cc:.4f}')
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_xlim(E_MIN, E_MAX)
    
    # Panel 3: Residual
    ax3 = axes[1, 0]
    residual = noisy_spectrum - recon_spectrum
    ax3.plot(energy, residual, 'k-', alpha=0.5, linewidth=0.5)
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.fill_between(energy, -2*np.std(residual), 2*np.std(residual), alpha=0.1, color='blue')
    ax3.set_xlabel('Energy (keV)')
    ax3.set_ylabel('Residual')
    ax3.set_title(f'(c) Fit Residual — RMSE={metrics.get("spectrum_RMSE", 0):.4f}')
    ax3.set_xlim(E_MIN, E_MAX)
    
    # Panel 4: Concentration comparison (bar chart)
    ax4 = axes[1, 1]
    x = np.arange(len(elements))
    width = 0.35
    gt_vals = [gt_concentrations[el] for el in elements]
    fit_vals = [refined.get(el, 0) for el in elements]
    
    bars1 = ax4.bar(x - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, fit_vals, width, label='Fitted', color='coral', alpha=0.8)
    
    # Add RE labels
    for i, el in enumerate(elements):
        re = metrics.get(f'{el}_RE_pct', 0)
        ax4.annotate(f'{re:.1f}%', (x[i] + width/2, fit_vals[i] + 2), ha='center', fontsize=7)
    
    ax4.set_xlabel('Element')
    ax4.set_ylabel('Concentration (a.u.)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(elements)
    mean_re = metrics.get('mean_RE_pct', 0)
    conc_cc = metrics.get('concentration_CC', 0)
    ax4.set_title(f'(d) Concentration Recovery — CC={conc_cc:.4f}, Mean RE={mean_re:.2f}%')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")

# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  pyxrf_fluor — XRF Fluorescence Spectrum Deconvolution")
    print("=" * 60)
    
    # (a) Generate data
    noisy_spectrum, gt_array, metadata = load_or_generate_data()
    energy = metadata['energy']
    clean_spectrum = metadata['clean_spectrum']
    
    print(f"[DATA] Spectrum shape: {noisy_spectrum.shape}")
    
    # (b) Run deconvolution
    recon_result = reconstruct(noisy_spectrum, metadata)
    
    # (c) Evaluate
    metrics = compute_metrics(GT_CONCENTRATIONS, recon_result, clean_spectrum,
                              recon_result['recon_spectrum'])
    
    print(f"\n[EVAL] === Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(energy, noisy_spectrum, clean_spectrum, recon_result,
                      GT_CONCENTRATIONS, metrics, vis_path)
    
    # (f) Save arrays
    gt_conc_array = np.array([GT_CONCENTRATIONS[el] for el in SAMPLE_ELEMENTS])
    fit_conc_array = np.array([recon_result['refined_concentrations'][el] for el in SAMPLE_ELEMENTS])
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_conc_array)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), fit_conc_array)
    
    print("=" * 60)
    print("  DONE — pyxrf_fluor XRF deconvolution complete")
    print("=" * 60)
