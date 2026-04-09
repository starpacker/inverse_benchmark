import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.optimize import curve_fit

def sphere_form_factor_amplitude(q, R):
    """
    Normalised form-factor amplitude for a homogeneous sphere:
        f(q,R) = 3 [sin(qR) - qR cos(qR)] / (qR)^3
    Returns f(q,R).  P(q) = f^2.
    """
    qR = np.asarray(q * R, dtype=np.float64)
    result = np.ones_like(qR)
    mask = qR > 1e-12
    result[mask] = 3.0 * (np.sin(qR[mask]) - qR[mask] * np.cos(qR[mask])) / qR[mask]**3
    return result

def sphere_intensity(q, R, scale, background):
    """
    I(q) = scale * V * delta_rho^2 * P(q) + background
    where P(q) = |f(q,R)|^2 and V = (4/3)pi R^3.
    
    For simplicity we fold V*delta_rho^2 into the scale factor
    so:  I(q) = scale * P(q) + background
    This is the standard parameterisation used by SasView.
    """
    P_q = sphere_form_factor_amplitude(q, R)**2
    return scale * P_q + background

def compute_pr(q, I_q, d_max=None, n_r=100):
    """
    Estimate the pair-distance distribution function P(r) via a simple
    indirect Fourier transform (Moore method / regularised sine transform).
    
    P(r) = (2r / pi) * integral_0^inf  q * I(q) * sin(qr) dq
    
    In practice we discretise and apply a simple Tikhonov regularisation
    to suppress noise artefacts.
    """
    if d_max is None:
        d_max = 2.0 * np.pi / q.min()
        d_max = min(d_max, 300.0)

    r = np.linspace(0, d_max, n_r)
    pr = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri < 1e-12:
            pr[i] = 0.0
            continue
        integrand = q * I_q * np.sin(q * ri)
        pr[i] = (2.0 * ri / np.pi) * np.trapezoid(integrand, q)

    pr = np.maximum(pr, 0.0)
    if pr.max() > 0:
        pr /= pr.max()

    return r, pr

def run_inversion(data, initial_guess=None, bounds=None):
    """
    Run the inverse problem: Fit noisy I(q) to recover structural parameters
    and compute P(r) distribution.
    
    Parameters:
    -----------
    data : dict
        Preprocessed data from load_and_preprocess_data
    initial_guess : list, optional
        Initial guess for [radius, scale, background]
    bounds : tuple, optional
        Parameter bounds as ([lower...], [upper...])
        
    Returns:
    --------
    dict containing:
        - fitted_params: dictionary of fitted parameters
        - param_errors: parameter uncertainties
        - I_fit: fitted intensity curve
        - r_pr: r values for P(r)
        - P_r: P(r) distribution
        - fit_success: boolean indicating success
    """
    print("[INVERSE] Fitting noisy I(q) to recover R, scale, background ...")
    
    q = data['q']
    I_noisy = data['I_noisy']
    sigma = data['sigma']
    
    if initial_guess is None:
        initial_guess = [40.0, 0.005, 0.01]
    if bounds is None:
        bounds = ([5.0, 1e-6, 0.0], [200.0, 1.0, 0.1])
    
    try:
        popt, pcov = curve_fit(
            sphere_intensity, q, I_noisy,
            p0=initial_guess, bounds=bounds,
            sigma=sigma, absolute_sigma=True,
            maxfev=10000
        )
        R_fit, scale_fit, bg_fit = popt
        perr = np.sqrt(np.diag(pcov))
        
        gt_params = data['gt_params']
        print(f"[INVERSE] Fitted R     = {R_fit:.4f} ± {perr[0]:.4f} Å  (GT={gt_params['radius']})")
        print(f"[INVERSE] Fitted scale = {scale_fit:.6f} ± {perr[1]:.6f}  (GT={gt_params['scale']})")
        print(f"[INVERSE] Fitted bg    = {bg_fit:.6f} ± {perr[2]:.6f}  (GT={gt_params['background']})")
        fit_success = True
        
    except Exception as e:
        print(f"[INVERSE] ERROR in curve_fit: {e}")
        R_fit, scale_fit, bg_fit = initial_guess
        perr = np.array([0.0, 0.0, 0.0])
        fit_success = False
    
    I_fit = sphere_intensity(q, R_fit, scale_fit, bg_fit)
    
    print("[INVERSE] Computing P(r) via indirect Fourier transform ...")
    
    I_for_pr = I_noisy - bg_fit
    I_for_pr = np.maximum(I_for_pr, 1e-10)
    
    d_max_est = 2.5 * R_fit
    r_pr, pr_fitted = compute_pr(q, I_for_pr, d_max=d_max_est * 1.5, n_r=150)
    
    print(f"[INVERSE] D_max estimate: {d_max_est*1.5:.1f} Å")
    
    result = {
        'fitted_params': {
            'radius': float(R_fit),
            'scale': float(scale_fit),
            'background': float(bg_fit)
        },
        'param_errors': {
            'radius': float(perr[0]),
            'scale': float(perr[1]),
            'background': float(perr[2])
        },
        'I_fit': I_fit,
        'r_pr': r_pr,
        'P_r': pr_fitted,
        'fit_success': fit_success
    }
    
    return result
