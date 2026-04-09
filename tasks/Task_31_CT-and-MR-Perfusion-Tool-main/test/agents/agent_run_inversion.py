import numpy as np

from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

from scipy.optimize import curve_fit

from scipy.linalg import toeplitz

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

def run_inversion(ctc_volumes, time_index, mask, ttp_indices, svd_thresh=0.1, rho=1.05, hcf=0.73):
    """
    Solves the inverse problem: Given Concentration C(t), find CBF, CBV, MTT.
    Involves:
    1. AIF Estimation (Blind deconvolution aspect).
    2. SVD Deconvolution (Numerical Inversion).
    """
    print("Running Inversion (SVD Deconvolution)...")
    
    # Convert TTP indices to physical time for AIF logic
    ttp_map = np.zeros_like(ttp_indices, dtype=float)
    # Handle mask
    valid_mask = mask > 0
    # Map indices to time
    ttp_vals_flat = np.array(time_index)[ttp_indices[valid_mask]]
    ttp_map[valid_mask] = ttp_vals_flat
    
    # 1. Determine AIF
    try:
        aif_props, aif_mask = _determine_aif_logic(ctc_volumes, time_index, mask, ttp_map)
        print(f"  AIF Parameters found: {aif_props}")
    except ValueError as e:
        print(f"  AIF detection failed: {e}. Using global max heuristic.")
        # Fallback: voxel with max AUC
        auc = ctc_volumes.sum(axis=0) * mask
        flat_idx = np.argmax(auc)
        z, y, x = np.unravel_index(flat_idx, auc.shape)
        aif_curve = ctc_volumes[:, z, y, x]
        aif_props = fit_gamma_variate(time_index, aif_curve)
        
    # Generate AIF curve
    aif_signal = gamma_variate(time_index, *aif_props)
    
    # 2. Construct Forward Matrix (Convolution Matrix) for SVD
    # We use block-circulant SVD formulation for robustness or standard Toeplitz
    dt = np.mean(np.diff(time_index))
    N = len(time_index)
    
    # Method: Standard Toeplitz for simplicity/robustness in this refactor
    # C = A * R * dt  ->  R = inv(A*dt) * C
    col = aif_signal
    row = np.zeros(N)
    row[0] = col[0]
    A_mat = toeplitz(col, row) * dt
    
    # 3. SVD Regularization
    U, S, Vt = np.linalg.svd(A_mat)
    
    # Truncate singular values
    max_s = np.max(S)
    inv_S = np.zeros_like(S)
    idx_keep = S >= (svd_thresh * max_s)
    inv_S[idx_keep] = 1.0 / S[idx_keep]
    
    # Construct pseudo-inverse
    A_inv = Vt.T @ np.diag(inv_S) @ U.T
    
    # 4. Apply Inversion to all voxels
    # Reshape Data: (T, Voxels)
    dims = ctc_volumes.shape # T, Z, Y, X
    n_voxels = np.prod(dims[1:])
    
    ctc_flat = ctc_volumes.reshape(dims[0], n_voxels)
    
    # R = A_inv * C
    residue_flat = np.dot(A_inv, ctc_flat)
    
    # 5. Extract Metrics from Residue Function R(t)
    # R(t) = CBF * R_ideal(t). Theoretically R_ideal(0)=1.
    # So max(R) approx CBF (scaled).
    
    # CBF = max(R) * 60 * 100 / (rho/hcf) (units)
    # CBV = integral(R) * ... or integral(C)/integral(AIF)
    
    # Reshape back
    residue_vol = residue_flat.reshape(dims)
    
    # Map Calculations
    # CBF: Peak of residue function
    cbf_raw = np.max(residue_vol, axis=0)
    cbf_map = (cbf_raw * 60 * 100 * hcf) / rho
    
    # CBV: Area under CTC / Area under AIF (More robust than integrating R)
    # But using R: CBV = sum(R) * dt * units
    # Let's use the standard SVD output definition:
    # CBV = CBF * MTT.
    # Alternatively: CBV = integral(C) / integral(AIF) is standard.
    auc_ctc = np.sum(ctc_volumes, axis=0)
    auc_aif = np.sum(aif_signal)
    
    # Check div zero
    safe_auc_aif = auc_aif if auc_aif > 0 else 1.0
    cbv_map = (auc_ctc / safe_auc_aif) * 100 * hcf / rho
    
    # MTT: CBV / CBF * 60
    mtt_map = np.zeros_like(cbf_map)
    with np.errstate(divide='ignore', invalid='ignore'):
        mtt_map = (cbv_map / cbf_map) * 60
        mtt_map[~np.isfinite(mtt_map)] = 0
        mtt_map[cbf_map == 0] = 0

    # Tmax: Time at which residue function is max? No, usually time C(t) is max is TTP.
    # Tmax in perfusion usually refers to the delay of the residue function peak 
    # relative to AIF peak, or simply argmax(R).
    tmax_indices = np.argmax(residue_vol, axis=0)
    tmax_map = np.array([time_index[i] for i in tmax_indices.flatten()]).reshape(tmax_indices.shape)
    
    # Apply mask
    results = {
        'CBF': cbf_map * mask,
        'CBV': cbv_map * mask,
        'MTT': mtt_map * mask,
        'TMAX': tmax_map * mask,
        'TTP': ttp_map * mask # Passed through
    }
    
    return results
