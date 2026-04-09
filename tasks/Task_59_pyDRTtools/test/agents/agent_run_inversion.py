import numpy as np

import matplotlib

matplotlib.use('Agg')

try:
    import pyDRTtools
    from pyDRTtools.runs import simple_run
    from pyDRTtools.basics import EIS_object
    HAS_PYDRTT = True
except ImportError:
    HAS_PYDRTT = False
    print("[WARN] pyDRTtools not found, using Tikhonov fallback")

HAS_PYDRTT = True

def forward_operator(gamma, tau, freq, r_inf, r_pol):
    """
    Compute EIS impedance from DRT via Fredholm integral.
    
    Z(ω) = R_∞ + R_pol ∫ γ(τ)/(1 + iωτ) d(ln τ)
    
    Parameters
    ----------
    gamma : np.ndarray
        DRT values γ(τ).
    tau : np.ndarray
        Relaxation times [s].
    freq : np.ndarray
        Frequencies [Hz].
    r_inf : float
        High-frequency resistance [Ω].
    r_pol : float
        Polarisation resistance [Ω].
    
    Returns
    -------
    Z : np.ndarray
        Complex impedance [Ω].
    """
    omega = 2 * np.pi * freq
    ln_tau = np.log(tau)
    
    # Compute d(ln τ) for integration
    d_ln_tau = np.zeros_like(ln_tau)
    d_ln_tau[1:-1] = (ln_tau[2:] - ln_tau[:-2]) / 2
    d_ln_tau[0] = ln_tau[1] - ln_tau[0]
    d_ln_tau[-1] = ln_tau[-1] - ln_tau[-2]
    
    Z = np.full(len(freq), r_inf, dtype=complex)
    for i, w in enumerate(omega):
        integrand = gamma / (1 + 1j * w * tau)
        Z[i] += r_pol * np.sum(integrand * d_ln_tau)
    
    return Z

def run_inversion(data_dict):
    """
    Recover DRT from noisy EIS data using Tikhonov regularization.
    
    Uses pyDRTtools if available (Tikhonov + GCV/L-curve),
    otherwise falls back to direct Tikhonov with scipy.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing:
        - 'freq': frequency array [Hz]
        - 'tau': relaxation time array [s]
        - 'Z_noisy': noisy impedance data
        - 'r_inf': high-frequency resistance
        - 'r_pol': polarisation resistance
    
    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - 'gamma_rec': recovered DRT
        - 'Z_fit': fitted impedance
        - All original data_dict entries
    """
    freq = data_dict['freq']
    tau = data_dict['tau']
    Z_noisy = data_dict['Z_noisy']
    r_inf = data_dict['r_inf']
    r_pol = data_dict['r_pol']
    
    n_freq = len(freq)
    n_tau = len(tau)
    
    gamma_rec = None
    Z_fit = None
    
    if HAS_PYDRTT:
        print("[RECON] Using pyDRTtools Tikhonov inversion ...")
        try:
            # Create EIS object for pyDRTtools
            eis = EIS_object(freq, Z_noisy.real, Z_noisy.imag)
            # Run DRT analysis
            result = simple_run(eis, rbf_type='Gaussian',
                               data_used='Combined Re-Im',
                               induct_used=0, der_used='1st',
                               lambda_value=1e-3,
                               NMC_sample=0)
            gamma_rec_raw = result.gamma
            tau_out = result.tau
            
            # Interpolate to our tau grid
            gamma_rec = np.interp(np.log10(tau), np.log10(tau_out), gamma_rec_raw)
            gamma_rec = np.maximum(gamma_rec, 0)
            
            # Compute fit
            Z_fit = forward_operator(gamma_rec, tau, freq, r_inf, r_pol)
        except Exception as e:
            print(f"[WARN] pyDRTtools failed: {e}, using fallback")
            gamma_rec = None
    
    # Fallback: Direct Tikhonov
    if gamma_rec is None:
        print("[RECON] Tikhonov DRT inversion (Fredholm integral) ...")
        
        omega = 2 * np.pi * freq
        ln_tau = np.log(tau)
        d_ln_tau = np.zeros_like(ln_tau)
        d_ln_tau[1:-1] = (ln_tau[2:] - ln_tau[:-2]) / 2
        d_ln_tau[0] = ln_tau[1] - ln_tau[0]
        d_ln_tau[-1] = ln_tau[-1] - ln_tau[-2]
        
        # Build kernel matrix A where Z = R_inf + R_pol * A @ gamma
        A = np.zeros((n_freq, n_tau), dtype=complex)
        for i, w in enumerate(omega):
            A[i, :] = r_pol * d_ln_tau / (1 + 1j * w * tau)
        
        # Stack real and imaginary parts
        A_stack = np.vstack([A.real, A.imag])
        b = np.hstack([
            Z_noisy.real - r_inf,
            Z_noisy.imag,
        ])
        
        # Smoothness matrix (first derivative)
        D = np.zeros((n_tau - 1, n_tau))
        for i in range(n_tau - 1):
            D[i, i] = -1
            D[i, i + 1] = 1
        
        # GCV for lambda selection
        lambdas = np.logspace(-6, 0, 20)
        gcv_scores = []
        for lam in lambdas:
            AtA = A_stack.T @ A_stack + lam * D.T @ D
            try:
                gamma_trial = np.linalg.solve(AtA, A_stack.T @ b)
            except np.linalg.LinAlgError:
                gcv_scores.append(1e20)
                continue
            gamma_trial = np.maximum(gamma_trial, 0)
            resid = b - A_stack @ gamma_trial
            n = len(b)
            try:
                H = A_stack @ np.linalg.solve(AtA, A_stack.T)
                trace_H = np.trace(H)
            except Exception:
                trace_H = n_tau
            gcv = (np.sum(resid ** 2) / n) / max((1 - trace_H / n) ** 2, 1e-12)
            gcv_scores.append(gcv)
        
        best_lam = lambdas[np.argmin(gcv_scores)]
        print(f"[RECON]   Best λ = {best_lam:.2e} (GCV)")
        
        AtA = A_stack.T @ A_stack + best_lam * D.T @ D
        gamma_rec = np.linalg.solve(AtA, A_stack.T @ b)
        gamma_rec = np.maximum(gamma_rec, 0)
        
        Z_fit = forward_operator(gamma_rec, tau, freq, r_inf, r_pol)
    
    # Build result dictionary
    result_dict = dict(data_dict)
    result_dict['gamma_rec'] = gamma_rec
    result_dict['Z_fit'] = Z_fit
    
    return result_dict
