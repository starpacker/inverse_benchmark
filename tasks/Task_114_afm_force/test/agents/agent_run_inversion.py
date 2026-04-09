import numpy as np

import matplotlib

matplotlib.use("Agg")

def run_inversion(z, delta_f, k, f0, A):
    """
    Recover force from frequency shift using the Sader-Jarvis formula:

    F(z) = 2k ∫_z^∞ [ (1 + A^{1/2}/(8√(π(t-z)))) Ω(t)
                       - A^{3/2}/√(2(t-z)) dΩ/dt ] dt
    where Ω(t) = Δf(t)/f0
    
    Parameters:
    -----------
    z : ndarray
        Distance grid (m)
    delta_f : ndarray
        Frequency shift (Hz)
    k : float
        Cantilever spring constant (N/m)
    f0 : float
        Resonance frequency (Hz)
    A : float
        Oscillation amplitude (m)
        
    Returns:
    --------
    F_recovered : ndarray
        Reconstructed force curve (N)
    """
    n = len(z)
    dz = z[1] - z[0]
    Omega = delta_f / f0
    dOmega = np.gradient(Omega, dz)
    
    F_recovered = np.zeros(n)
    
    for i in range(n - 3):
        # Integration from z[i] to z[-1]
        t = z[i+1:]
        Om_t = Omega[i+1:]
        dOm_t = dOmega[i+1:]
        dt_val = t - z[i]
        
        # Regularize singularity
        dt_safe = np.maximum(dt_val, dz * 0.1)
        
        # Sader-Jarvis integrand with regularization
        sqrt_term = np.sqrt(A) / (8.0 * np.sqrt(np.pi * dt_safe))
        # Limit the singular correction terms
        sqrt_term = np.minimum(sqrt_term, 10.0)
        
        term1 = (1.0 + sqrt_term) * Om_t
        
        deriv_term = A**1.5 / np.sqrt(2.0 * dt_safe)
        deriv_term = np.minimum(deriv_term, 100.0 * np.max(np.abs(Om_t)))
        term2 = -deriv_term * dOm_t
        
        integrand = term1 + term2
        
        F_recovered[i] = 2.0 * k * np.trapezoid(integrand, t)
    
    # Extrapolate last points
    for i in range(n-3, n):
        F_recovered[i] = F_recovered[n-4]
    
    return F_recovered
