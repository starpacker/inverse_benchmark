import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import minimize, differential_evolution

def forward_operator(params_vec, freq):
    """
    Compute complex impedance of a Randles circuit with Warburg element.
    
    Circuit: R0-p(R1,C1)-W (Randles circuit)
        R0 = series/ohmic resistance
        R1 = charge transfer resistance
        C1 = double-layer capacitance
        sigma_W = Warburg coefficient
    
    Z(ω) = R0 + Z_RC(ω) + Z_W(ω)
    where Z_RC = R1 / (1 + jωR1C1)   (parallel RC)
          Z_W  = σ_W / sqrt(ω) * (1 - j)  (Warburg impedance)
    
    Parameters
    ----------
    params_vec : ndarray or list
        [R0, R1, C1, sigma_W] circuit parameters.
    freq : ndarray
        Frequency array in Hz.
    
    Returns
    -------
    Z : ndarray (complex)
        Complex impedance at each frequency.
    """
    R0, R1, C1, sigma_W = params_vec
    omega = 2.0 * np.pi * freq
    
    # Parallel RC element: Z_RC = R1 / (1 + j*omega*R1*C1)
    Z_RC = R1 / (1.0 + 1j * omega * R1 * C1)
    
    # Warburg element: Z_W = sigma_W / sqrt(omega) * (1 - j)
    Z_W = sigma_W / np.sqrt(omega) * (1.0 - 1j)
    
    # Total impedance
    Z = R0 + Z_RC + Z_W
    
    return Z

def run_inversion(data, bounds, initial_guesses):
    """
    Run nonlinear least squares optimization to fit circuit parameters.
    
    Uses multiple starting points with L-BFGS-B and differential evolution
    to avoid local minima.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data.
    bounds : list of tuples
        Parameter bounds [(low, high), ...] for each parameter.
    initial_guesses : list of ndarray
        List of initial parameter guesses.
    
    Returns
    -------
    result : dict
        Contains fitted_params, Z_fitted, best_cost, converged, optimization_result.
    """
    freq = data['freq']
    Z_meas = data['Z_noisy']
    
    def objective(params_vec):
        """Weighted least-squares objective on real + imaginary parts."""
        Z_model = forward_operator(params_vec, freq)
        # Weight by 1/|Z_meas| so all frequencies contribute equally
        w = 1.0 / np.abs(Z_meas)
        residual_re = (Z_model.real - Z_meas.real) * w
        residual_im = (Z_model.imag - Z_meas.imag) * w
        return 0.5 * np.sum(residual_re**2 + residual_im**2)
    
    best_result = None
    best_cost = np.inf
    
    # Try multiple starting points with L-BFGS-B
    for x0 in initial_guesses:
        res = minimize(objective, x0,
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-14})
        if res.fun < best_cost:
            best_cost = res.fun
            best_result = res
    
    # Refine with differential evolution (derivative-free, good for noisy landscapes)
    de_result = differential_evolution(objective, bounds,
                                        seed=42, maxiter=2000, tol=1e-14,
                                        polish=True, workers=1)
    if de_result.fun < best_cost:
        best_cost = de_result.fun
        best_result = de_result
    
    fitted_params = best_result.x
    Z_fitted = forward_operator(fitted_params, freq)
    
    print("Optimization converged:", best_result.success)
    print(f"Objective value: {best_result.fun:.6e}")
    
    result = {
        'fitted_params': fitted_params,
        'Z_fitted': Z_fitted,
        'best_cost': best_cost,
        'converged': bool(best_result.success),
        'optimization_result': best_result,
    }
    
    return result
