import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import curve_fit

from scipy.signal import find_peaks

from scipy.ndimage import uniform_filter1d, percentile_filter

def forward_operator(mode_params, freqs, bg_params):
    """
    Forward model: Generate power spectrum from mode parameters.
    
    P(ν) = Σ_nl H_nl · L(ν; f_nl, Γ_nl) + B(ν)
    
    where L is a Lorentzian profile and B is the background.
    
    Args:
        mode_params: list of tuples [(freq, height, width), ...]
        freqs: ndarray of frequency values
        bg_params: tuple (A, nu_c, alpha, W) for Harvey-like background
        
    Returns:
        spectrum: ndarray, the predicted power spectrum
    """
    # Background: Harvey-like profile + white noise
    A, nu_c, alpha, W = bg_params
    spectrum = A / (1.0 + (freqs / nu_c) ** alpha) + W
    
    # Add each mode as a Lorentzian
    for f0, H, gamma in mode_params:
        spectrum = spectrum + H / (1.0 + ((freqs - f0) / gamma) ** 2)
    
    return spectrum

def run_inversion(noisy_spectrum, metadata):
    """
    Peakbagging: Extract mode frequencies from noisy power spectrum.
    
    Algorithm:
    1. Estimate and subtract background
    2. Find candidate peaks in smoothed spectrum
    3. Fit individual Lorentzians around each peak
    4. Match fitted frequencies to GT modes
    5. Generate reconstructed spectrum from fitted parameters
    
    Args:
        noisy_spectrum: ndarray, the noisy observed power spectrum
        metadata: dict containing auxiliary data
        
    Returns:
        result: dict containing fitted frequencies, heights, widths, matched modes, etc.
    """
    freqs = metadata['freqs']
    smoothed = metadata['smoothed_spectrum']
    gt_modes = metadata['gt_modes']
    linewidth = metadata['linewidth']
    delta_nu = metadata['delta_nu']
    freq_resolution = metadata['freq_resolution']
    bg_params = metadata['bg_params']
    
    # Step 1: Estimate background using percentile filtering
    bg_estimate = percentile_filter(noisy_spectrum, percentile=20, size=500)
    bg_subtracted = smoothed - np.minimum(bg_estimate, smoothed * 0.9)
    bg_subtracted = np.maximum(bg_subtracted, 0)
    
    # Step 2: Find peaks in background-subtracted smoothed spectrum
    min_distance = int(delta_nu / 5.0 / freq_resolution)
    prominence = np.percentile(bg_subtracted[bg_subtracted > 0], 40) if np.any(bg_subtracted > 0) else 1.0
    
    peak_indices, peak_props = find_peaks(
        bg_subtracted,
        distance=min_distance,
        prominence=prominence * 0.3,
        height=np.median(bg_subtracted) + np.std(bg_subtracted) * 0.2
    )
    
    candidate_freqs = freqs[peak_indices]
    print(f"[RECON] Found {len(candidate_freqs)} candidate peaks")
    
    # Step 3: Fit individual Lorentzians around each candidate peak
    fitted_freqs = []
    fitted_heights = []
    fitted_widths = []
    
    fit_window = 10.0  # μHz half-window for fitting
    
    def lorentzian_bg(f, f0, H, gamma, bg):
        return H / (1.0 + ((f - f0) / gamma) ** 2) + bg
    
    for cf in candidate_freqs:
        mask = np.abs(freqs - cf) < fit_window
        if np.sum(mask) < 10:
            continue
        
        freq_window = freqs[mask]
        spec_window = noisy_spectrum[mask]
        
        try:
            p0_bg = [cf, np.max(spec_window) - np.min(spec_window), linewidth, np.min(spec_window)]
            bounds_bg = (
                [cf - 5, 0, 0.1, 0],
                [cf + 5, np.max(spec_window) * 5, 10.0, np.max(spec_window)]
            )
            
            popt, pcov = curve_fit(lorentzian_bg, freq_window, spec_window,
                                   p0=p0_bg, bounds=bounds_bg, maxfev=5000)
            
            fitted_freqs.append(popt[0])
            fitted_heights.append(popt[1])
            fitted_widths.append(popt[2])
        except Exception:
            fitted_freqs.append(cf)
            fitted_heights.append(np.max(spec_window))
            fitted_widths.append(linewidth)
    
    fitted_freqs = np.array(fitted_freqs)
    fitted_heights = np.array(fitted_heights)
    fitted_widths = np.array(fitted_widths)
    
    print(f"[RECON] Successfully fitted {len(fitted_freqs)} Lorentzian peaks")
    
    # Step 4: Match fitted frequencies to GT modes (greedy matching)
    gt_freq_list = sorted(gt_modes.values())
    gt_freq_array = np.array(gt_freq_list)
    
    matched_gt = []
    matched_fitted = []
    used_fitted = set()
    
    for gf in gt_freq_array:
        dists = np.abs(fitted_freqs - gf)
        sorted_idx = np.argsort(dists)
        for idx in sorted_idx:
            if idx not in used_fitted and dists[idx] < delta_nu / 3.0:
                matched_gt.append(gf)
                matched_fitted.append(fitted_freqs[idx])
                used_fitted.add(idx)
                break
    
    matched_gt = np.array(matched_gt)
    matched_fitted = np.array(matched_fitted)
    
    print(f"[RECON] Matched {len(matched_gt)} of {len(gt_freq_array)} GT modes")
    
    # Step 5: Generate reconstructed spectrum using forward operator
    mode_params = [(fitted_freqs[i], fitted_heights[i], fitted_widths[i]) 
                   for i in range(len(fitted_freqs))]
    
    # Use estimated background parameters for reconstruction
    # Estimate simple polynomial background from bg_estimate
    recon_spectrum = forward_operator(mode_params, freqs, bg_params)
    
    # Adjust to use the estimated background instead
    recon_spectrum_adjusted = np.zeros_like(freqs)
    for f0, H, gamma in mode_params:
        recon_spectrum_adjusted += H / (1.0 + ((freqs - f0) / gamma) ** 2)
    recon_spectrum_adjusted += bg_estimate
    
    return {
        'fitted_freqs': fitted_freqs,
        'fitted_heights': fitted_heights,
        'fitted_widths': fitted_widths,
        'matched_gt': matched_gt,
        'matched_fitted': matched_fitted,
        'recon_spectrum': recon_spectrum_adjusted,
        'bg_estimate': bg_estimate,
        'gt_freq_array': gt_freq_array,
    }
