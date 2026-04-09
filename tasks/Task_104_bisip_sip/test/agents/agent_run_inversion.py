import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

import time

from scipy.optimize import minimize

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_104_bisip_sip"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def forward_operator(freq, rho0, m, tau, c):
    """
    Cole-Cole complex resistivity forward model.
    ρ*(ω) = ρ_0 × [1 - m × (1 - 1/(1 + (iωτ)^c))]
    
    Parameters:
        freq: ndarray, frequency array (Hz)
        rho0: float, DC resistivity (Ohm·m)
        m: float, chargeability (0-1)
        tau: float, time constant (s)
        c: float, frequency exponent (0-1)
    
    Returns:
        rho_star: ndarray, complex resistivity
    """
    omega = 2.0 * np.pi * freq
    z = (1j * omega * tau) ** c
    rho_star = rho0 * (1.0 - m * (1.0 - 1.0 / (1.0 + z)))
    return rho_star

def run_inversion(freq, rho_obs_list, rho_true_list, gt_params_list):
    """
    Run Cole-Cole model inversion for multiple spectra using multi-start
    nonlinear least squares optimization.
    
    Parameters:
        freq: ndarray, frequency array
        rho_obs_list: list of ndarray, observed complex resistivity for each spectrum
        rho_true_list: list of ndarray, true complex resistivity for each spectrum
        gt_params_list: list of dicts, ground truth parameters
    
    Returns:
        results: list of dicts containing inversion results for each spectrum
    """
    
    def objective(params_vec, freq, rho_obs):
        """Least-squares objective for Cole-Cole inversion."""
        rho0, m, tau, c = params_vec
        
        # Enforce bounds implicitly
        if rho0 <= 0 or m <= 0 or m >= 1 or tau <= 0 or c <= 0 or c >= 1:
            return 1e10
        
        rho_model = forward_operator(freq, rho0, m, tau, c)
        
        # Normalized misfit (amplitude + phase)
        amp_obs = np.abs(rho_obs)
        amp_mod = np.abs(rho_model)
        phase_obs = np.angle(rho_obs)
        phase_mod = np.angle(rho_model)
        
        misfit_amp = np.sum(((amp_obs - amp_mod) / amp_obs) ** 2)
        misfit_phase = np.sum((phase_obs - phase_mod) ** 2)
        
        return misfit_amp + misfit_phase
    
    results = []
    
    for idx, (rho_obs, rho_true, gt_p) in enumerate(zip(rho_obs_list, rho_true_list, gt_params_list)):
        print(f"\n--- Spectrum {idx + 1}/{len(gt_params_list)} ---")
        print(f"    GT: ρ₀={gt_p['rho0']}, m={gt_p['m']}, τ={gt_p['tau']}, c={gt_p['c']}")
        
        t0 = time.time()
        
        # Multi-start optimization
        best_result = None
        best_cost = np.inf
        
        rho0_guesses = [50.0, 150.0, 300.0, 500.0]
        m_guesses = [0.2, 0.4, 0.6]
        tau_guesses = [0.01, 0.1, 1.0, 10.0]
        c_guesses = [0.3, 0.5, 0.7]
        
        bounds = [(1.0, 1000.0), (0.01, 0.99), (1e-4, 100.0), (0.05, 0.95)]
        
        for rho0_init in rho0_guesses:
            for m_init in m_guesses:
                for tau_init in tau_guesses:
                    for c_init in c_guesses:
                        x0 = [rho0_init, m_init, tau_init, c_init]
                        try:
                            result = minimize(
                                objective, x0, args=(freq, rho_obs),
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 500, 'ftol': 1e-12}
                            )
                            if result.fun < best_cost:
                                best_cost = result.fun
                                best_result = result
                        except Exception:
                            continue
        
        if best_result is not None:
            rho0, m, tau, c = best_result.x
            fit_p = {"rho0": rho0, "m": m, "tau": tau, "c": c}
        else:
            fit_p = {"rho0": 100.0, "m": 0.3, "tau": 0.1, "c": 0.5}
        
        elapsed = time.time() - t0
        print(f"    Fit: ρ₀={fit_p['rho0']:.2f}, m={fit_p['m']:.4f}, "
              f"τ={fit_p['tau']:.4f}, c={fit_p['c']:.4f}  ({elapsed:.1f}s)")
        
        # Compute fit spectrum using forward operator
        rho_fit = forward_operator(freq, fit_p["rho0"], fit_p["m"], fit_p["tau"], fit_p["c"])
        
        results.append({
            "gt": gt_p,
            "fit": fit_p,
            "rho_true": rho_true,
            "rho_obs": rho_obs,
            "rho_fit": rho_fit,
        })
    
    return results
