import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

import radvel

def load_and_preprocess_data(true_params, n_obs, t_span, t_start, rv_err_val, seed=42):
    """
    Generate synthetic radial velocity observations.
    
    Parameters
    ----------
    true_params : dict
        Dictionary containing true orbital parameters
    n_obs : int
        Number of observations
    t_span : float
        Observation time span (days)
    t_start : float
        Start time (JD)
    rv_err_val : float
        Measurement uncertainty (m/s)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    t : ndarray
        Observation times
    rv_obs : ndarray
        Observed radial velocities
    rv_err : ndarray
        Measurement uncertainties
    rv_true : ndarray
        True radial velocities (without noise)
    true_params : dict
        True parameters (passed through for convenience)
    """
    np.random.seed(seed)
    
    # Irregular time sampling (simulating real observing cadence)
    t = np.sort(t_start + t_span * np.random.rand(n_obs))
    
    # Add some observing gaps (simulating weather, telescope scheduling)
    mask = ~((t > t_start + 60) & (t < t_start + 75))  # 15-day gap
    mask &= ~((t > t_start + 180) & (t < t_start + 190))  # 10-day gap
    t = t[mask]
    
    # Compute true RV using forward operator
    rv_true = forward_operator(t, true_params)
    
    # Add noise (measurement error + jitter)
    total_err = np.sqrt(rv_err_val**2 + true_params['jit']**2)
    rv_err = np.ones(len(t)) * rv_err_val  # formal errors
    noise = total_err * np.random.randn(len(t))
    rv_obs = rv_true + noise
    
    print(f"  Generated {len(t)} observations over {t[-1]-t[0]:.1f} days")
    print(f"  RV range: [{rv_obs.min():.1f}, {rv_obs.max():.1f}] m/s")
    print(f"  True jitter: {true_params['jit']:.1f} m/s, measurement σ: {rv_err_val:.1f} m/s")
    
    return t, rv_obs, rv_err, rv_true, true_params

def forward_operator(t, params_dict):
    """
    Forward: Compute RV time series from orbital parameters.
    
    For each planet: v_i(t) = K_i [cos(ν_i(t) + ω_i) + e_i cos(ω_i)]
    Total: v(t) = Σ v_i(t) + γ
    
    Parameters
    ----------
    t : ndarray
        Time array (JD)
    params_dict : dict
        Dictionary containing orbital parameters:
        - per1, tp1, e1, w1, k1: Planet 1 parameters
        - per2, tp2, e2, w2, k2: Planet 2 parameters
        - gamma: Systemic velocity
        
    Returns
    -------
    rv_total : ndarray
        Predicted radial velocity at each time
    """
    rv_total = np.zeros_like(t, dtype=np.float64)
    
    # Planet 1
    if params_dict.get('k1', 0) != 0:
        orbel1 = np.array([
            params_dict['per1'],
            params_dict['tp1'],
            params_dict['e1'],
            params_dict['w1'],
            params_dict['k1']
        ])
        rv_total += radvel.kepler.rv_drive(t, orbel1)
    
    # Planet 2
    if params_dict.get('k2', 0) != 0:
        orbel2 = np.array([
            params_dict['per2'],
            params_dict['tp2'],
            params_dict['e2'],
            params_dict['w2'],
            params_dict['k2']
        ])
        rv_total += radvel.kepler.rv_drive(t, orbel2)
    
    # Systemic velocity
    rv_total += params_dict.get('gamma', 0.0)
    
    return rv_total
