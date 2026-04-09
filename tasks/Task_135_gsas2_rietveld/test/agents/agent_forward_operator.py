import numpy as np

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
