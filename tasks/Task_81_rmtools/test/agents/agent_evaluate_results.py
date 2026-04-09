import numpy as np

import matplotlib

matplotlib.use('Agg')

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

def evaluate_results(ground_truth, reconstruction, components, d_phi):
    """
    Compute Faraday depth spectrum recovery metrics.
    
    Following standard radio interferometry practice, the GT spectrum is
    convolved with a Gaussian 'CLEAN beam' whose FWHM matches the RMSF
    before comparison. This is analogous to comparing a CLEAN image with
    the model convolved with the restoring beam.
    
    Args:
        ground_truth: dict with Q_clean, U_clean, components
        reconstruction: dict with phi_arr, cleanFDF, dirtyFDF, fwhmRMSF
        components: list of dicts with 'phi', 'amplitude', 'chi0'
        d_phi: Faraday depth resolution
    
    Returns:
        dict with psnr, rmse, cc, avg_peak_position_error, peak_position_errors,
             avg_amplitude_recovery, fwhm_rmsf, gt_peaks, clean_peaks
    """
    phi_arr = reconstruction['phi_arr']
    cleanFDF = reconstruction['cleanFDF']
    fwhmRMSF = reconstruction['fwhmRMSF']
    
    # Generate GT Faraday depth spectrum on same grid
    gt_FDF = generate_faraday_depth_spectrum(phi_arr, components, d_phi)
    
    # Convolve GT amplitude with CLEAN beam (Gaussian with FWHM = RMSF FWHM)
    # sigma_beam in pixel units: FWHM / (2*sqrt(2*ln2)) / D_PHI
    sigma_beam_pix = fwhmRMSF / (2.0 * np.sqrt(2.0 * np.log(2.0))) / d_phi
    gt_amp_raw = np.abs(gt_FDF)
    gt_amp = gaussian_filter1d(gt_amp_raw, sigma_beam_pix)
    
    # Use absolute values (amplitude spectrum)
    clean_amp = np.abs(cleanFDF)
    dirty_amp = np.abs(reconstruction['dirtyFDF'])
    
    # Normalize for comparison
    gt_amp_norm = gt_amp / (gt_amp.max() + 1e-10)
    clean_amp_norm = clean_amp / (clean_amp.max() + 1e-10)
    
    # RMSE
    rmse = np.sqrt(np.mean((gt_amp_norm - clean_amp_norm)**2))
    
    # Correlation coefficient
    cc = np.corrcoef(gt_amp_norm, clean_amp_norm)[0, 1]
    
    # Peak detection accuracy
    gt_peaks = []
    clean_peaks = []
    for comp in components:
        phi0 = comp['phi']
        # Find GT peak
        idx_gt = np.argmin(np.abs(phi_arr - phi0))
        gt_peaks.append(phi_arr[idx_gt])
        
        # Find clean peak within ±fwhmRMSF of true position
        fwhm = reconstruction['fwhmRMSF']
        mask = np.abs(phi_arr - phi0) < 2 * fwhm
        if np.any(mask):
            idx_clean = np.argmax(clean_amp[mask])
            clean_peaks.append(phi_arr[mask][idx_clean])
        else:
            clean_peaks.append(phi0)  # fallback
    
    # Peak position errors
    peak_errors = [abs(g - c) for g, c in zip(gt_peaks, clean_peaks)]
    avg_peak_error = np.mean(peak_errors)
    
    # Amplitude recovery at peak positions
    amp_recoveries = []
    for comp in components:
        phi0 = comp['phi']
        fwhm = reconstruction['fwhmRMSF']
        mask = np.abs(phi_arr - phi0) < fwhm
        if np.any(mask):
            peak_amp = np.max(clean_amp[mask])
            amp_recoveries.append(peak_amp / comp['amplitude'])
    avg_amp_recovery = np.mean(amp_recoveries) if amp_recoveries else 0.0
    
    # PSNR
    data_range = gt_amp_norm.max() - gt_amp_norm.min()
    mse = np.mean((gt_amp_norm - clean_amp_norm)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    return {
        'psnr': float(psnr),
        'rmse': float(rmse),
        'cc': float(cc),
        'avg_peak_position_error': float(avg_peak_error),
        'peak_position_errors': [float(e) for e in peak_errors],
        'avg_amplitude_recovery': float(avg_amp_recovery),
        'fwhm_rmsf': float(reconstruction['fwhmRMSF']),
        'gt_peaks': [float(p) for p in gt_peaks],
        'clean_peaks': [float(p) for p in clean_peaks],
    }
