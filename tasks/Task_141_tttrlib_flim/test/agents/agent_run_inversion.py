import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy.optimize import differential_evolution

from scipy.signal import fftconvolve

def forward_operator(
    time: np.ndarray,
    irf: np.ndarray,
    a1: float,
    tau1: float,
    a2: float,
    tau2: float,
    bg: float,
    total_counts: int
) -> np.ndarray:
    """
    Compute the forward model: IRF convolved with bi-exponential decay plus background.
    
    F(t) = IRF(t) ⊛ [ a₁·exp(-t/τ₁) + a₂·exp(-t/τ₂) ] + background
    
    Parameters
    ----------
    time : np.ndarray
        Time axis (ns).
    irf : np.ndarray
        Normalized instrument response function.
    a1 : float
        Amplitude for first exponential component.
    tau1 : float
        Lifetime for first component (ns).
    a2 : float
        Amplitude for second exponential component.
    tau2 : float
        Lifetime for second component (ns).
    bg : float
        Background counts per bin.
    total_counts : int
        Total photon counts for normalization.
        
    Returns
    -------
    np.ndarray
        Model prediction (counts per bin).
    """
    # Bi-exponential decay
    decay = a1 * np.exp(-time / tau1) + a2 * np.exp(-time / tau2)
    
    # Convolve with IRF
    convolved = fftconvolve(irf, decay, mode="full")[:len(time)]
    
    # Normalize to total counts
    convolved = convolved / convolved.sum() * total_counts
    
    # Add background
    convolved += bg
    
    return convolved

def run_inversion(
    time: np.ndarray,
    irf: np.ndarray,
    measured: np.ndarray,
    total_counts: int,
    bounds: list,
    rng_seed: int,
    maxiter: int = 2000,
    tol: float = 1e-10
) -> dict:
    """
    Run differential evolution optimization to recover decay parameters.
    
    Minimizes reduced chi-squared between measured data and forward model.
    
    Parameters
    ----------
    time : np.ndarray
        Time axis (ns).
    irf : np.ndarray
        Normalized instrument response function.
    measured : np.ndarray
        Measured TCSPC histogram (counts).
    total_counts : int
        Total photon counts for normalization.
    bounds : list
        Parameter bounds: [(a1_min, a1_max), (tau1_min, tau1_max), 
                          (tau2_min, tau2_max), (bg_min, bg_max)].
    rng_seed : int
        Random seed for optimizer.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'a1_fit': fitted amplitude 1
        - 'a2_fit': fitted amplitude 2
        - 'tau1_fit': fitted lifetime 1 (ns)
        - 'tau2_fit': fitted lifetime 2 (ns)
        - 'bg_fit': fitted background
        - 'fitted_curve': fitted model curve
        - 'reduced_chi2': reduced chi-squared value
        - 'success': optimization convergence flag
    """
    
    def objective(params):
        """Reduced chi-squared cost function (Poisson weighting)."""
        a1, tau1, tau2, bg = params
        a2 = 1.0 - a1  # constrain amplitudes to sum to 1
        
        model = forward_operator(time, irf, a1, tau1, a2, tau2, bg, total_counts)
        
        # Poisson variance ≈ max(model, 1) to avoid division by zero
        variance = np.maximum(model, 1.0)
        chi2 = np.sum((measured - model) ** 2 / variance)
        n_dof = len(measured) - len(params)
        return chi2 / n_dof
    
    result = differential_evolution(
        objective,
        bounds,
        seed=rng_seed,
        maxiter=maxiter,
        tol=tol,
        polish=True,
        workers=1,
    )
    
    # Extract fitted parameters
    a1_fit, tau1_fit, tau2_fit, bg_fit = result.x
    a2_fit = 1.0 - a1_fit
    
    # Ensure tau1 < tau2 (swap if needed for consistency)
    if tau1_fit > tau2_fit:
        tau1_fit, tau2_fit = tau2_fit, tau1_fit
        a1_fit, a2_fit = a2_fit, a1_fit
    
    # Compute fitted curve
    fitted_curve = forward_operator(
        time, irf, a1_fit, tau1_fit, a2_fit, tau2_fit, bg_fit, total_counts
    )
    
    return {
        'a1_fit': a1_fit,
        'a2_fit': a2_fit,
        'tau1_fit': tau1_fit,
        'tau2_fit': tau2_fit,
        'bg_fit': bg_fit,
        'fitted_curve': fitted_curve,
        'reduced_chi2': result.fun,
        'success': result.success,
    }
