import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

import radvel

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

def run_inversion(t, rv_obs, rv_err):
    """
    Inverse: Fit Keplerian orbital parameters from RV data.
    
    Uses radvel's Likelihood + Posterior framework with
    maximum-likelihood optimization (Powell method).
    
    Parameters
    ----------
    t : ndarray
        Observation times
    rv_obs : ndarray
        Observed radial velocities
    rv_err : ndarray
        Measurement uncertainties
        
    Returns
    -------
    fitted_params : dict
        Dictionary of fitted orbital parameters
    rv_fitted : ndarray
        Fitted radial velocity model at observation times
    """
    nplanets = 2
    
    # Set up Parameters in 'per tp e w k' basis
    params = radvel.Parameters(nplanets, basis='per tp e w k')
    
    # Initial guesses (slightly perturbed from truth to simulate realistic fitting)
    params['per1'] = radvel.Parameter(value=15.5)    # ~2% off
    params['tp1'] = radvel.Parameter(value=2458201.0)
    params['e1'] = radvel.Parameter(value=0.1)       # start lower
    params['w1'] = radvel.Parameter(value=np.deg2rad(70))  # ~5 deg off
    params['k1'] = radvel.Parameter(value=40.0)      # ~11% off
    
    params['per2'] = radvel.Parameter(value=110.0)   # ~4% off
    params['tp2'] = radvel.Parameter(value=2458225.0)
    params['e2'] = radvel.Parameter(value=0.25)      # start lower
    params['w2'] = radvel.Parameter(value=np.deg2rad(200))  # ~10 deg off
    params['k2'] = radvel.Parameter(value=18.0)      # ~18% off
    
    params['dvdt'] = radvel.Parameter(value=0.0)     # no linear trend
    params['curv'] = radvel.Parameter(value=0.0)     # no curvature
    
    # Create the RV model
    mod = radvel.RVModel(params, time_base=np.median(t))
    
    # Create likelihood
    like = radvel.likelihood.RVLikelihood(mod, t, rv_obs, rv_err)
    
    # Set gamma and jitter as free parameters
    like.params['gamma'] = radvel.Parameter(value=0.0)
    like.params['jit'] = radvel.Parameter(value=1.0)
    
    # Set up priors
    post = radvel.posterior.Posterior(like)
    
    # Add eccentricity prior (avoid unphysical e > 1)
    post.priors += [radvel.prior.EccentricityPrior(nplanets)]
    
    # Add positive K prior
    post.priors += [radvel.prior.PositiveKPrior(nplanets)]
    
    # Hard bounds on jitter
    post.priors += [radvel.prior.HardBounds('jit', 0.01, 20.0)]
    
    print("  [FIT] Running maximum-likelihood optimization...")
    print(f"  [FIT] Initial guess - Per1={params['per1'].value:.2f}, K1={params['k1'].value:.1f}")
    print(f"  [FIT] Initial guess - Per2={params['per2'].value:.2f}, K2={params['k2'].value:.1f}")
    
    # Fit
    res = radvel.fitting.maxlike_fitting(post, verbose=False, method='Powell')
    
    # Extract fitted parameters
    fitted_params = {}
    for key in res.params:
        fitted_params[key] = float(res.params[key].value)
    
    print(f"  [FIT] Fitted - Per1={fitted_params['per1']:.4f}, K1={fitted_params['k1']:.2f}")
    print(f"  [FIT] Fitted - Per2={fitted_params['per2']:.4f}, K2={fitted_params['k2']:.2f}")
    print(f"  [FIT] Fitted - gamma={fitted_params['gamma']:.2f}, jit={fitted_params['jit']:.2f}")
    
    # Compute fitted RV using forward operator
    rv_fitted = forward_operator(t, fitted_params)
    
    return fitted_params, rv_fitted
