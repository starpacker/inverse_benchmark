import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.special import voigt_profile

from lmfit import Parameters, minimize

def gaussian(x, amplitude, center, sigma):
    """Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def voigt(x, amplitude, center, sigma, gamma):
    """Voigt peak – convolution of Gaussian and Lorentzian."""
    vp = voigt_profile(x - center, sigma, gamma)
    vp_max = voigt_profile(0.0, sigma, gamma)
    if vp_max > 0:
        return amplitude * vp / vp_max
    return np.zeros_like(x)

def forward_operator(x, params, peak_defs):
    """
    Evaluate the composite spectral model given parameters.
    
    This is the forward model: parameters -> predicted spectrum
    
    Args:
        x: numpy array of x-axis values
        params: lmfit Parameters object or dict-like with parameter values
        peak_defs: list of dicts defining peak types
    
    Returns:
        y_pred: numpy array of predicted spectrum values
    """
    # Get baseline parameters
    if hasattr(params, '__getitem__'):
        # lmfit Parameters object
        baseline_intercept = params['baseline_intercept'].value if hasattr(params['baseline_intercept'], 'value') else params['baseline_intercept']
        baseline_slope = params['baseline_slope'].value if hasattr(params['baseline_slope'], 'value') else params['baseline_slope']
    else:
        baseline_intercept = params['baseline_intercept']
        baseline_slope = params['baseline_slope']
    
    y = baseline_intercept + baseline_slope * x
    
    for i, pk in enumerate(peak_defs):
        if hasattr(params, '__getitem__'):
            amp_param = params[f'p{i}_amplitude']
            cen_param = params[f'p{i}_center']
            amp = amp_param.value if hasattr(amp_param, 'value') else amp_param
            cen = cen_param.value if hasattr(cen_param, 'value') else cen_param
        else:
            amp = params[f'p{i}_amplitude']
            cen = params[f'p{i}_center']
        
        if pk["type"] == "gaussian":
            if hasattr(params, '__getitem__'):
                sig_param = params[f'p{i}_sigma']
                sig = sig_param.value if hasattr(sig_param, 'value') else sig_param
            else:
                sig = params[f'p{i}_sigma']
            y = y + gaussian(x, amp, cen, sig)
            
        elif pk["type"] == "lorentzian":
            if hasattr(params, '__getitem__'):
                gam_param = params[f'p{i}_gamma']
                gam = gam_param.value if hasattr(gam_param, 'value') else gam_param
            else:
                gam = params[f'p{i}_gamma']
            y = y + lorentzian(x, amp, cen, gam)
            
        elif pk["type"] == "voigt":
            if hasattr(params, '__getitem__'):
                sig_param = params[f'p{i}_sigma']
                gam_param = params[f'p{i}_gamma']
                sig = sig_param.value if hasattr(sig_param, 'value') else sig_param
                gam = gam_param.value if hasattr(gam_param, 'value') else gam_param
            else:
                sig = params[f'p{i}_sigma']
                gam = params[f'p{i}_gamma']
            y = y + voigt(x, amp, cen, sig, gam)
    
    return y

def run_inversion(x, measured_spectrum, true_peaks, perturbation_scale=0.12, seed=42):
    """
    Run the spectral peak fitting inversion.
    
    Args:
        x: numpy array of x-axis values
        measured_spectrum: numpy array of measured (noisy) spectrum
        true_peaks: list of dicts with peak type info (used for model structure)
        perturbation_scale: fractional perturbation for initial guesses
        seed: random seed for reproducibility
    
    Returns:
        fitted_spectrum: numpy array of fitted spectrum
        result: lmfit MinimizerResult object
        fitted_baseline: numpy array of fitted baseline
        individual_peaks_fit: list of numpy arrays for each fitted peak
    """
    np.random.seed(seed)
    
    print("[RECON] Setting up lmfit composite model ...")
    
    def residual_func(params, x_data, data, peak_defs):
        """Residual = data - model."""
        return data - forward_operator(x_data, params, peak_defs)
    
    # Initialize parameters with perturbed values
    params = Parameters()
    params.add('baseline_intercept', value=0.3, min=-5, max=5)
    params.add('baseline_slope', value=0.0008, min=-0.01, max=0.01)
    
    for i, pk in enumerate(true_peaks):
        amp_init = pk["amplitude"] * (1 + perturbation_scale * np.random.randn())
        cen_init = pk["center"] + perturbation_scale * 20 * np.random.randn()
        params.add(f'p{i}_amplitude', value=max(amp_init, 0.1), min=0.01, max=50)
        params.add(f'p{i}_center', value=cen_init, min=pk["center"] - 80, max=pk["center"] + 80)
        
        if pk["type"] in ("gaussian", "voigt"):
            sig_init = pk["sigma"] * (1 + perturbation_scale * np.random.randn())
            params.add(f'p{i}_sigma', value=max(sig_init, 1.0), min=0.5, max=100)
        
        if pk["type"] in ("lorentzian", "voigt"):
            gam_init = pk["gamma"] * (1 + perturbation_scale * np.random.randn())
            params.add(f'p{i}_gamma', value=max(gam_init, 1.0), min=0.5, max=100)
    
    print("[RECON] Running least-squares optimization ...")
    result = minimize(residual_func, params, args=(x, measured_spectrum, true_peaks),
                      method='leastsq', max_nfev=10000)
    
    print(f"[RECON]   Fit converged: {result.success}")
    print(f"[RECON]   Num function evals: {result.nfev}")
    print(f"[RECON]   Reduced chi-square: {result.redchi:.6f}")
    
    # Extract fitted spectrum using forward operator
    fitted_spectrum = forward_operator(x, result.params, true_peaks)
    
    # Extract fitted baseline
    fitted_baseline = result.params['baseline_intercept'].value + result.params['baseline_slope'].value * x
    
    # Extract individual fitted peaks
    individual_peaks_fit = []
    for i, pk in enumerate(true_peaks):
        amp = result.params[f'p{i}_amplitude'].value
        cen = result.params[f'p{i}_center'].value
        
        if pk["type"] == "gaussian":
            sig = result.params[f'p{i}_sigma'].value
            y = gaussian(x, amp, cen, sig)
        elif pk["type"] == "lorentzian":
            gam = result.params[f'p{i}_gamma'].value
            y = lorentzian(x, amp, cen, gam)
        elif pk["type"] == "voigt":
            sig = result.params[f'p{i}_sigma'].value
            gam = result.params[f'p{i}_gamma'].value
            y = voigt(x, amp, cen, sig, gam)
        else:
            y = np.zeros_like(x)
        
        individual_peaks_fit.append(y)
    
    print("[RECON] Fitting complete.")
    
    return fitted_spectrum, result, fitted_baseline, individual_peaks_fit
