import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.optimize import curve_fit

E_CHARGE = 1.602176634e-19

M_ELECTRON = 9.1093837015e-31

K_BOLTZMANN = 1.380649e-23

EV_TO_K = E_CHARGE / K_BOLTZMANN

A_PROBE = 1.0e-6

def electron_saturation_current(T_e_eV, n_e):
    """Electron saturation current [A] for given T_e (eV) and n_e (m⁻³)."""
    T_e_K = T_e_eV * EV_TO_K
    return n_e * E_CHARGE * A_PROBE * np.sqrt(K_BOLTZMANN * T_e_K / (2 * np.pi * M_ELECTRON))

def floating_potential(T_e_eV, n_e, V_p, I_ion_sat):
    """Compute floating potential V_f where I(V_f) = 0."""
    I_e_sat = electron_saturation_current(T_e_eV, n_e)
    if I_e_sat <= 0 or -I_ion_sat <= 0:
        return V_p  # degenerate
    T_e_K = T_e_eV * EV_TO_K
    V_f = V_p + (K_BOLTZMANN * T_e_K / E_CHARGE) * np.log(-I_ion_sat / I_e_sat)
    return V_f

def relative_error(true_val, est_val):
    """Compute relative error between true and estimated values."""
    return abs(est_val - true_val) / abs(true_val) if true_val != 0 else 0.0

def forward_operator(V, T_e, n_e, V_p, I_ion_sat):
    """
    Theoretical Langmuir probe I-V characteristic (forward model).
    
    Parameters
    ----------
    V : array_like
        Bias voltage [V]
    T_e : float
        Electron temperature [eV]
    n_e : float
        Electron density [m⁻³]
    V_p : float
        Plasma potential [V]
    I_ion_sat : float
        Ion saturation current [A] (negative)
    
    Returns
    -------
    I : ndarray
        Probe current [A]
    """
    T_e_K = T_e * EV_TO_K
    I_e_sat = electron_saturation_current(T_e, n_e)
    
    # Clamp the exponent to avoid overflow
    exponent = E_CHARGE * (V - V_p) / (K_BOLTZMANN * T_e_K)
    exponent = np.clip(exponent, -500, 500)
    
    I = np.where(
        V < V_p,
        I_ion_sat + I_e_sat * np.exp(exponent),
        I_ion_sat + I_e_sat,
    )
    return I

def run_inversion(preprocessed_data):
    """
    Run inversion on all preprocessed data cases.
    
    Parameters
    ----------
    preprocessed_data : list of dict
        Output from load_and_preprocess_data
    
    Returns
    -------
    results : list of dict
        Each dict contains:
        - 'case': case label
        - 'true_params': dict of true parameters
        - 'fitted_params': dict of fitted parameters
        - 'V_f_true': true floating potential
        - 'relative_errors': dict of relative errors
        - 'V': voltage array
        - 'I_clean': clean current array
        - 'I_noisy': noisy current array
        - 'I_fitted': fitted current array
        - 'std_errors': dict of standard errors
    """
    results = []
    
    for data in preprocessed_data:
        V = data['V']
        I_noisy = data['I_noisy']
        I_clean = data['I_clean']
        true_params = data['true_params']
        label = data['label']
        
        # Heuristic initial guesses
        I_min = np.min(I_noisy)
        I_max = np.max(I_noisy)
        V_p_guess = V[np.argmax(np.gradient(I_noisy, V))]
        T_e_guess = 5.0
        n_e_guess = 1e17
        I_ion_sat_guess = I_min
        p0 = [T_e_guess, n_e_guess, V_p_guess, I_ion_sat_guess]
        
        bounds = (
            [0.1, 1e14, V.min(), -1.0],    # lower
            [100.0, 1e20, V.max(), 0.0],   # upper
        )
        
        popt, pcov = curve_fit(
            forward_operator, V, I_noisy,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
            method="trf",
        )
        
        T_e_fit, n_e_fit, V_p_fit, I_ion_sat_fit = popt
        V_f_fit = floating_potential(T_e_fit, n_e_fit, V_p_fit, I_ion_sat_fit)
        perr = np.sqrt(np.diag(pcov))
        
        fitted_params = {
            "T_e": T_e_fit,
            "n_e": n_e_fit,
            "V_p": V_p_fit,
            "I_ion_sat": I_ion_sat_fit,
            "V_f": V_f_fit,
        }
        
        std_errors = {
            "T_e": perr[0],
            "n_e": perr[1],
            "V_p": perr[2],
            "I_ion_sat": perr[3],
        }
        
        # Compute fitted curve
        I_fitted = forward_operator(V, T_e_fit, n_e_fit, V_p_fit, I_ion_sat_fit)
        
        # Compute relative errors
        re = {
            "T_e": relative_error(true_params["T_e"], fitted_params["T_e"]),
            "n_e": relative_error(true_params["n_e"], fitted_params["n_e"]),
            "V_p": relative_error(true_params["V_p"], fitted_params["V_p"]),
            "I_ion_sat": relative_error(true_params["I_ion_sat"], fitted_params["I_ion_sat"]),
            "V_f": relative_error(true_params["V_f"], fitted_params["V_f"]),
        }
        
        results.append({
            "case": label,
            "true_params": true_params,
            "fitted_params": fitted_params,
            "V_f_true": true_params["V_f"],
            "relative_errors": re,
            "V": V,
            "I_clean": I_clean,
            "I_noisy": I_noisy,
            "I_fitted": I_fitted,
            "std_errors": std_errors,
        })
    
    return results
