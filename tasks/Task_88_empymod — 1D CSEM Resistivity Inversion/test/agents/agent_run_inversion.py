import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

from scipy.optimize import differential_evolution

def forward_operator(depth, res, frequencies, offsets, src_depth_abs, rec_depth_abs):
    """
    Compute CSEM frequency-domain EM response using empymod.
    
    Args:
        depth: List of layer interface depths
        res: List of layer resistivities
        frequencies: Array of frequencies (Hz)
        offsets: Array of source-receiver offsets (m)
        src_depth_abs: Absolute source depth (m)
        rec_depth_abs: Absolute receiver depth (m)
    
    Returns:
        response: Complex E-field array of shape (n_freq, n_off)
    """
    import empymod
    
    n_freq = len(frequencies)
    n_off = len(offsets)
    response = np.zeros((n_freq, n_off), dtype=complex)
    
    for j, offset in enumerate(offsets):
        resp = empymod.dipole(
            src=[0, 0, src_depth_abs],
            rec=[offset, 0, rec_depth_abs],
            depth=depth,
            res=res,
            freqtime=frequencies,
            verb=0,
        )
        response[:, j] = resp
    
    return response

def run_inversion(data_dict, n_iter=25, lambda_reg=10.0):
    """
    CSEM inversion using differential evolution (global) + L-BFGS-B (polish).
    Works in log10-resistivity space for better conditioning.
    Uses normalized log-amplitude misfit for robust gradient across decades.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data
        n_iter: Maximum iterations (used for maxiter in DE)
        lambda_reg: Regularization parameter (not directly used in DE)
    
    Returns:
        result_dict: Dictionary containing inversion results
    """
    data_obs = data_dict['data_obs']
    depth = data_dict['depth']
    res_init = data_dict['res_init']
    frequencies = data_dict['frequencies']
    offsets = data_dict['offsets']
    src_z = data_dict['src_z']
    rec_z = data_dict['rec_z']
    param_indices = data_dict['param_indices']
    
    # Observed data: log-amplitude (normalized) + phase
    obs_flat = data_obs.flatten()
    log_amp_obs = np.log10(np.abs(obs_flat) + 1e-30)
    phase_obs = np.angle(obs_flat)
    
    # Normalization: scale log-amplitude to unit range
    amp_range = log_amp_obs.max() - log_amp_obs.min()
    log_amp_obs_norm = (log_amp_obs - log_amp_obs.min()) / amp_range
    log_amp_obs_min = log_amp_obs.min()
    
    errors = []
    call_count = [0]
    
    def objective(m_log10):
        res_current = list(res_init)
        for k, idx in enumerate(param_indices):
            res_current[idx] = 10**m_log10[k]
        
        E_pred = forward_operator(depth, res_current, frequencies, offsets, src_z, rec_z)
        pred_flat = E_pred.flatten()
        
        # Log-amplitude misfit (normalized)
        log_amp_pred = np.log10(np.abs(pred_flat) + 1e-30)
        log_amp_pred_norm = (log_amp_pred - log_amp_obs_min) / amp_range
        
        misfit_amp = np.mean((log_amp_obs_norm - log_amp_pred_norm)**2)
        
        # Phase misfit (using sin for wrapping)
        misfit_phase = np.mean(np.sin(phase_obs - np.angle(pred_flat))**2)
        
        # Combined objective (amplitude-dominated for CSEM)
        total = misfit_amp + 0.3 * misfit_phase
        
        call_count[0] += 1
        if call_count[0] % 50 == 0:
            errors.append(total)
            res_vals = [10**m_log10[k] for k in range(len(param_indices))]
            print(f"  Eval {call_count[0]}: obj={total:.6f} "
                  f"(amp={misfit_amp:.6f}, phase={misfit_phase:.6f}) "
                  f"ρ={[f'{v:.2f}' for v in res_vals]}")
        
        return total
    
    # Bounds in log10: overburden [0.3-5], reservoir [5-200], basement [0.3-10]
    bounds_log10 = [
        (-0.5, 0.8),   # Overburden: 0.3 - 6 Ωm (sediment)
        (0.5, 2.5),    # Reservoir: 3 - 300 Ωm (hydrocarbon)
        (-0.5, 1.2),   # Basement: 0.3 - 16 Ωm (crystalline rock)
    ]
    
    print(f"  Running differential evolution (global optimization)...")
    result = differential_evolution(
        objective,
        bounds=bounds_log10,
        seed=42,
        maxiter=200,
        tol=1e-10,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,       # L-BFGS-B polishing at the end
        disp=False,
    )
    
    # Extract final model
    res_inv = list(res_init)
    for k, idx in enumerate(param_indices):
        res_inv[idx] = 10**result.x[k]
    
    print(f"  Optimization converged: {result.success}, message: {result.message}")
    print(f"  Total evaluations: {result.nfev}")
    print(f"  Final resistivities: {[res_inv[i] for i in param_indices]}")
    
    # Compute forward response of inverted model
    data_inv = forward_operator(depth, res_inv, frequencies, offsets, src_z, rec_z)
    
    result_dict = {
        'res_inv': res_inv,
        'data_inv': data_inv,
        'errors': errors,
        'optimization_result': result,
    }
    
    return result_dict
