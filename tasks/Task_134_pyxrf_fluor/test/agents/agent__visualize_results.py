import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def gaussian_peak(energy, center, amplitude, sigma):
    """Gaussian peak profile for detector-broadened XRF line."""
    return amplitude * np.exp(-0.5 * ((energy - center) / sigma) ** 2)

def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def generate_element_spectrum(energy, element, concentration, det_sigma, xrf_lines):
    """
    Generate XRF spectrum for a single element.
    Each characteristic line is a Gaussian broadened by detector resolution.
    """
    spectrum = np.zeros_like(energy)
    if element not in xrf_lines:
        return spectrum
    
    for line_name, line_energy, rel_intensity in xrf_lines[element]:
        amplitude = concentration * rel_intensity
        spectrum += gaussian_peak(energy, line_energy, amplitude, det_sigma)
    
    return spectrum

def _visualize_results(energy, noisy_spectrum, clean_spectrum, recon_result, 
                       gt_concentrations, metrics, save_path, e_min, e_max,
                       detector_fwhm, xrf_lines):
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
    det_sigma = fwhm_to_sigma(detector_fwhm)
    for i, el in enumerate(elements):
        el_spec = generate_element_spectrum(energy, el, gt_concentrations[el], det_sigma, xrf_lines)
        ax1.fill_between(energy, 0, el_spec, alpha=0.3, color=colors[i], label=f'{el} (GT)')
    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Intensity (counts)')
    ax1.set_title('(a) GT XRF Spectrum — Element Contributions')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_xlim(e_min, e_max)
    
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
    ax2.set_xlim(e_min, e_max)
    
    # Panel 3: Residual
    ax3 = axes[1, 0]
    residual = noisy_spectrum - recon_spectrum
    ax3.plot(energy, residual, 'k-', alpha=0.5, linewidth=0.5)
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.fill_between(energy, -2*np.std(residual), 2*np.std(residual), alpha=0.1, color='blue')
    ax3.set_xlabel('Energy (keV)')
    ax3.set_ylabel('Residual')
    ax3.set_title(f'(c) Fit Residual — RMSE={metrics.get("spectrum_RMSE", 0):.4f}')
    ax3.set_xlim(e_min, e_max)
    
    # Panel 4: Concentration comparison (bar chart)
    ax4 = axes[1, 1]
    x = np.arange(len(elements))
    width = 0.35
    gt_vals = [gt_concentrations[el] for el in elements]
    fit_vals = [refined.get(el, 0) for el in elements]
    
    ax4.bar(x - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, fit_vals, width, label='Fitted', color='coral', alpha=0.8)
    
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
