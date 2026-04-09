import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

def generate_faraday_depth_spectrum(phi_arr, components, d_phi):
    """
    Generate ground truth Faraday depth spectrum F(φ) on a given φ grid.
    Each component is a delta function convolved with a narrow Gaussian
    for numerical representation.
    """
    F_gt = np.zeros(len(phi_arr), dtype=complex)
    
    for comp in components:
        phi0 = comp['phi']
        amp = comp['amplitude']
        chi0 = comp['chi0']
        
        # Represent as narrow Gaussian (delta-like)
        sigma_phi = d_phi * 0.5
        gaussian = amp * np.exp(-(phi_arr - phi0)**2 / (2 * sigma_phi**2))
        phase = np.exp(2j * chi0)
        F_gt += gaussian * phase
    
    return F_gt

def visualize_results(observations, ground_truth, reconstruction, metrics, save_path, components, d_phi):
    """Generate comprehensive RM synthesis visualization."""
    phi_arr = reconstruction['phi_arr']
    dirtyFDF = reconstruction['dirtyFDF']
    cleanFDF = reconstruction['cleanFDF']
    phi2_arr = reconstruction['phi2_arr']
    RMSFArr = reconstruction['RMSFArr']
    
    # GT spectrum — convolve with CLEAN beam for fair comparison
    gt_FDF_raw = generate_faraday_depth_spectrum(phi_arr, components, d_phi)
    fwhmRMSF = reconstruction['fwhmRMSF']
    sigma_beam_pix = fwhmRMSF / (2.0 * np.sqrt(2.0 * np.log(2.0))) / d_phi
    gt_FDF_amp = gaussian_filter1d(np.abs(gt_FDF_raw), sigma_beam_pix)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Input Q and U spectra
    freq_ghz = observations['freq_hz'] / 1e9
    axes[0, 0].plot(freq_ghz, observations['Q'], 'b.', ms=1, alpha=0.3, label='Q(ν)')
    axes[0, 0].plot(freq_ghz, observations['U'], 'r.', ms=1, alpha=0.3, label='U(ν)')
    axes[0, 0].plot(freq_ghz, ground_truth['Q_clean'], 'b-', lw=1, alpha=0.7)
    axes[0, 0].plot(freq_ghz, ground_truth['U_clean'], 'r-', lw=1, alpha=0.7)
    axes[0, 0].set_xlabel('Frequency (GHz)')
    axes[0, 0].set_ylabel('Flux (Jy/beam)')
    axes[0, 0].set_title('Stokes Q, U Spectra')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) RMSF
    center = len(phi2_arr) // 2
    width = min(200, len(phi2_arr) // 2)
    rmsf_slice = slice(center - width, center + width)
    axes[0, 1].plot(phi2_arr[rmsf_slice], np.abs(RMSFArr[rmsf_slice]), 'k-', lw=1.5)
    axes[0, 1].set_xlabel('φ (rad/m²)')
    axes[0, 1].set_ylabel('|RMSF|')
    axes[0, 1].set_title(f'RMSF (FWHM={metrics["fwhm_rmsf"]:.1f} rad/m²)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) Dirty FDF amplitude
    axes[0, 2].plot(phi_arr, np.abs(dirtyFDF), 'gray', lw=1, label='Dirty |F(φ)|')
    axes[0, 2].plot(phi_arr, gt_FDF_amp, 'b-', lw=2, label='GT |F(φ)| (convolved)')
    for comp in components:
        axes[0, 2].axvline(x=comp['phi'], color='b', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('φ (rad/m²)')
    axes[0, 2].set_ylabel('|F(φ)| (Jy/beam/RMSF)')
    axes[0, 2].set_title('Dirty Faraday Dispersion Function')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # (d) Clean FDF vs GT
    axes[1, 0].plot(phi_arr, gt_FDF_amp, 'b-', lw=2, label='GT |F(φ)| (convolved)')
    axes[1, 0].plot(phi_arr, np.abs(cleanFDF), 'r-', lw=1.5, label='CLEAN |F(φ)|')
    for i, comp in enumerate(components):
        axes[1, 0].axvline(x=comp['phi'], color='b', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=metrics['clean_peaks'][i], color='r', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('φ (rad/m²)')
    axes[1, 0].set_ylabel('|F(φ)| (Jy/beam/RMSF)')
    axes[1, 0].set_title(f'CLEAN FDF (CC={metrics["cc"]:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (e) Zoom on components
    for i, comp in enumerate(components):
        phi0 = comp['phi']
        fwhm = metrics['fwhm_rmsf']
        zoom_range = (-fwhm * 5, fwhm * 5)
        mask = (phi_arr > phi0 + zoom_range[0]) & (phi_arr < phi0 + zoom_range[1])
        
        ax = axes[1, 1] if i == 0 else axes[1, 2]
        ax.plot(phi_arr[mask], gt_FDF_amp[mask], 'b-', lw=2, label='GT (convolved)')
        ax.plot(phi_arr[mask], np.abs(cleanFDF[mask]), 'r--', lw=2, label='CLEAN')
        ax.plot(phi_arr[mask], np.abs(dirtyFDF[mask]), 'gray', lw=1, alpha=0.5, label='Dirty')
        ax.axvline(x=phi0, color='b', linestyle='--', alpha=0.5, label=f'True φ={phi0}')
        ax.set_xlabel('φ (rad/m²)')
        ax.set_ylabel('|F(φ)|')
        ax.set_title(f'Component {i+1}: φ={phi0}, Δφ={metrics["peak_position_errors"][i]:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"RM-Tools — Faraday Rotation Measure Synthesis & CLEAN\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC={metrics['cc']:.4f} | "
        f"Avg Peak Error={metrics['avg_peak_position_error']:.2f} rad/m² | "
        f"Amp Recovery={metrics['avg_amplitude_recovery']:.3f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")
