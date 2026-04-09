import numpy as np

from scipy.optimize import least_squares

import matplotlib

matplotlib.use('Agg')

def forward_operator(tt, params, wavelength, refs, raw_norm, tt_min, tt_max):
    """
    Compute the powder diffraction pattern given parameters.
    
    Parameters:
    -----------
    tt : np.ndarray
        2-theta array in degrees
    params : np.ndarray
        Parameters [a, scale, U, V, W, eta, bg0, bg1, bg2, bg3]
    wavelength : float
        X-ray wavelength in Angstroms
    refs : list
        List of reflection tuples (q2, F2, mult)
    raw_norm : float
        Normalization factor for intensities
    tt_min : float
        Minimum 2-theta for background normalization
    tt_max : float
        Maximum 2-theta for background normalization
    
    Returns:
    --------
    y : np.ndarray
        Calculated intensity pattern
    """
    a = params[0]
    scale = params[1]
    U = max(params[2], 1e-4)
    V = params[3]
    W = max(params[4], 1e-4)
    eta = np.clip(params[5], 0.01, 0.99)
    bg = params[6:]
    
    # Chebyshev background
    x = 2*(tt - tt_min)/(tt_max - tt_min) - 1
    y = bg[0]*np.ones_like(x)
    if len(bg) > 1:
        y = y + bg[1]*x
    if len(bg) > 2:
        y = y + bg[2]*(2*x*x - 1)
    if len(bg) > 3:
        y = y + bg[3]*(4*x*x*x - 3*x)
    
    target_max = 10000.0  # normalize so scale=1 → max peak ~ target_max
    
    for q2, F2, mult in refs:
        d = a / np.sqrt(q2)
        s = wavelength / (2*d)
        if abs(s) >= 1:
            continue
        tt_pk = 2*np.degrees(np.arcsin(s))
        if tt_pk < tt_min - 0.5 or tt_pk > tt_max + 0.5:
            continue
        th = np.radians(tt_pk/2)
        lp = (1 + np.cos(2*th)**2) / (np.sin(th)**2 * np.cos(th))
        dw = np.exp(-0.5*(np.sin(th)/wavelength)**2)
        
        raw_int = mult * F2 * lp * dw
        intensity = scale * target_max * raw_int / raw_norm
        
        # FWHM from Caglioti
        H = np.sqrt(max(U*np.tan(th)**2 + V*np.tan(th) + W, 1e-6))
        
        # Pseudo-Voigt profile
        sig = H / (2*np.sqrt(2*np.log(2)))
        gam = H / 2
        dx = tt - tt_pk
        gauss = np.exp(-0.5*(dx/sig)**2) / (sig*np.sqrt(2*np.pi))
        lorentz = (gam/np.pi) / (dx*dx + gam*gam)
        y = y + intensity * (eta*lorentz + (1-eta)*gauss)
    
    return y

def run_inversion(data):
    """
    Run the Rietveld refinement (inversion) using weighted least-squares.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data from load_and_preprocess_data
    
    Returns:
    --------
    dict containing:
        - refined_params: optimized parameters
        - y_calc: calculated pattern with refined parameters
        - result: scipy optimization result object
        - converged: boolean indicating convergence
        - n_evals: number of function evaluations
    """
    tt = data['tt']
    y_obs = data['y_obs']
    weights = data['weights']
    initial_params = data['initial_params']
    wavelength = data['wavelength']
    refs = data['refs']
    raw_norm = data['raw_norm']
    tt_min = data['tt_min']
    tt_max = data['tt_max']
    
    def residuals(p, tt, yobs, w):
        y_calc = forward_operator(tt, p, wavelength, refs, raw_norm, tt_min, tt_max)
        return w * (yobs - y_calc)
    
    # Parameter bounds
    lb = [4.5, 0.01, 0.0001, -0.1, 0.0001, 0.01, -200, -200, -200, -200]
    ub = [6.5, 10.0, 0.5,    0.1,  0.5,    0.99, 200,  200,  200,  200]
    
    # Run optimization
    result = least_squares(
        residuals, 
        initial_params, 
        args=(tt, y_obs, weights),
        bounds=(lb, ub), 
        method='trf',
        ftol=1e-12, 
        xtol=1e-12, 
        gtol=1e-12,
        max_nfev=5000, 
        verbose=1
    )
    
    refined_params = result.x
    y_calc = forward_operator(tt, refined_params, wavelength, refs, raw_norm, tt_min, tt_max)
    
    return {
        'refined_params': refined_params,
        'y_calc': y_calc,
        'result': result,
        'converged': bool(result.success),
        'n_evals': int(result.nfev)
    }
