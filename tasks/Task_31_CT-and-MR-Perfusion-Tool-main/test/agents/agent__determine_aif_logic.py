import numpy as np

from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

from scipy.optimize import curve_fit

from skimage import measure

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

def gamma_variate(t, t0, alpha, beta, amplitude=1.0):
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    result = np.zeros_like(t_shifted)
    mask = t > t0
    # Add small epsilon to avoid log(0) or div/0 issues implicitly
    safe_t = t_shifted[mask]
    result[mask] = amplitude * (safe_t**alpha) * np.exp(-safe_t/beta)
    return result

def fit_gamma_variate(time_index, curve):
    try:
        # bounds: [t0, alpha, beta, amp]
        # constrained to be physically plausible
        popt, _ = curve_fit(
            gamma_variate, 
            time_index, 
            curve, 
            bounds=([0, 0.1, 0.1, 0], [20, 8, 8, np.max(curve)*2.5]),
            maxfev=2000
        )
        return popt
    except Exception:
        return None

def calculate_signal_smoothness(signal):
    if len(signal) < 3: return float('inf')
    rnge = np.max(signal) - np.min(signal)
    if rnge == 0: return 0.0
    norm_sig = (signal - np.min(signal)) / rnge
    second_deriv = np.diff(norm_sig, n=2)
    return np.sum(second_deriv ** 2) / len(signal)

def _determine_aif_logic(ctc_volumes, time_index, mask, ttp, dilate_r=5, erode_r=2):
    # Adaptive thresholding logic
    ctc_stack = np.stack(ctc_volumes, axis=0)
    
    # Pre-calculate stats on valid mask
    valid_mask = mask > 0
    if not np.any(valid_mask):
        raise ValueError("Empty mask provided for AIF determination.")

    auc_map = ctc_stack.sum(axis=0)
    auc_vals = auc_map[valid_mask]
    ttp_vals = ttp[valid_mask]
    
    # Search grid
    percentiles = range(95, 4, -10)
    smoothness_t = 0.04
    
    for p in percentiles:
        attempt_auc = np.percentile(auc_vals, p)
        attempt_ttp = np.percentile(ttp_vals, 100 - p)
        
        # Initial candidate mask
        cand_mask = (auc_map > attempt_auc) & (ttp < attempt_ttp) & (ttp > 0)
        
        # Morphological cleanup
        d_k = np.ones((1, dilate_r, dilate_r), bool)
        e_k = np.ones((1, erode_r, erode_r), bool)
        cand_mask = binary_erosion(binary_dilation(cand_mask, structure=d_k), structure=e_k)
        
        labeled, n_comps = measure.label(cand_mask, return_num=True, connectivity=3)
        if n_comps == 0: continue
        
        candidates = []
        for i in range(1, n_comps + 1):
            comp_mask = (labeled == i)
            vol_size = np.sum(comp_mask)
            if not (5 < vol_size < 50): continue # Volume constraints
            
            # Extract curve
            curve_data = ctc_stack[:, comp_mask]
            avg_curve = curve_data.mean(axis=1)
            
            smooth = calculate_signal_smoothness(avg_curve)
            if smooth > smoothness_t: continue
            
            # Fit Gamma
            props = fit_gamma_variate(time_index, avg_curve)
            if props is None: continue
            
            # Calc error
            fitted = gamma_variate(time_index, *props)
            err = np.mean((avg_curve - fitted)**2)
            peak_diff = np.max(avg_curve) - np.mean(avg_curve[-3:])
            
            candidates.append({
                'idx': i,
                'vol': vol_size,
                'props': props,
                'error': err,
                'peak_diff': peak_diff,
                'score': (vol_size * peak_diff) / (err + 1e-6)
            })
            
        if candidates:
            # Select best
            df = pd.DataFrame(candidates)
            best = df.loc[df['score'].idxmax()]
            return best['props'], (labeled == best['idx'])
            
    # Fallback if sophisticated search fails: center of mass of high AUC?
    # For strict compliance, raise error if fails
    raise ValueError("Could not determine AIF from data.")
