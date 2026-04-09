import numpy as np

from scipy.interpolate import interp1d

def run_inversion(ccf_ref: np.ndarray, ccf_matrix: np.ndarray,
                  t: np.ndarray, stretch_range: float,
                  stretch_steps: int, coda_tmin: float) -> dict:
    """
    Stretching method inversion: find epsilon that maximises correlation
    between stretched reference and current CCF within the coda window.
    
    Parameters:
        ccf_ref: reference cross-correlation function
        ccf_matrix: matrix of perturbed CCFs (n_days x n_samples)
        t: time axis array
        stretch_range: trial stretching range (e.g., 0.01 for ±1%)
        stretch_steps: number of trial epsilon values
        coda_tmin: minimum absolute time for coda window
    
    Returns:
        Dictionary containing:
            - dvv_est: estimated dv/v values for each day
            - cc_best: best correlation coefficient for each day
    """
    n_days = ccf_matrix.shape[0]
    
    # Coda window mask: |t| >= coda_tmin
    coda_mask = np.abs(t) >= coda_tmin
    
    # Trial epsilon values
    trial_eps = np.linspace(-stretch_range, stretch_range, stretch_steps)
    
    # Interpolation function for reference CCF
    interp_func = interp1d(t, ccf_ref, kind='cubic',
                           bounds_error=False, fill_value=0.0)
    
    dvv_est = np.empty(n_days)
    cc_best_arr = np.empty(n_days)
    
    for d in range(n_days):
        ccf_cur = ccf_matrix[d]
        cc_values = np.empty(stretch_steps)
        
        # Current CCF coda window
        cur_coda = ccf_cur[coda_mask]
        cur_coda_demean = cur_coda - cur_coda.mean()
        cur_norm = np.sqrt(np.sum(cur_coda_demean ** 2))
        
        for i, eps in enumerate(trial_eps):
            t_stretched = t * (1.0 + eps)
            ref_stretched = interp_func(t_stretched)
            ref_coda = ref_stretched[coda_mask]
            ref_coda_demean = ref_coda - ref_coda.mean()
            ref_norm = np.sqrt(np.sum(ref_coda_demean ** 2))
            
            if ref_norm < 1e-15 or cur_norm < 1e-15:
                cc_values[i] = 0.0
            else:
                cc_values[i] = np.sum(ref_coda_demean * cur_coda_demean) / (
                    ref_norm * cur_norm)
        
        best_idx = np.argmax(cc_values)
        eps_best = trial_eps[best_idx]
        dvv_est[d] = -eps_best
        cc_best_arr[d] = cc_values[best_idx]
    
    return {
        "dvv_est": dvv_est,
        "cc_best": cc_best_arr,
    }
