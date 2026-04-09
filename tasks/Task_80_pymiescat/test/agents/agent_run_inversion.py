import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import minimize, differential_evolution

def forward_operator(n_real, k_imag, diameter, wavelengths):
    """
    Compute Mie scattering/absorption efficiency spectra.
    
    Forward model: y = A(x) where
        x = (n, k, d) - refractive index and diameter
        y = (Qsca(λ), Qabs(λ)) - efficiency spectra
    
    Uses Lorenz-Mie theory via PyMieScatt.MieQ
    
    Args:
        n_real: Real part of refractive index
        k_imag: Imaginary part of refractive index  
        diameter: Particle diameter (nm)
        wavelengths: Array of wavelengths (nm)
    
    Returns:
        qsca: Scattering efficiency spectrum (numpy array)
        qabs: Absorption efficiency spectrum (numpy array)
        qext: Extinction efficiency spectrum (numpy array)
        g_param: Asymmetry parameter spectrum (numpy array)
    """
    import PyMieScatt as ps
    
    m = complex(n_real, k_imag)
    num_wl = len(wavelengths)
    qsca = np.zeros(num_wl)
    qabs = np.zeros(num_wl)
    qext = np.zeros(num_wl)
    g_param = np.zeros(num_wl)
    
    for i, wl in enumerate(wavelengths):
        result = ps.MieQ(m, wl, diameter)
        # MieQ returns: (Qext, Qsca, Qabs, g, Qpr, Qback, Qratio)
        qext[i] = float(result[0])
        qsca[i] = float(result[1])
        qabs[i] = float(result[2])
        g_param[i] = float(result[3])
    
    return qsca, qabs, qext, g_param

def run_inversion(observations, n_bounds, k_bounds, d_bounds, seed=42):
    """
    Inverse problem: recover (n, k, d) from observed Mie spectra.
    
    Uses a two-stage approach:
    1. Global search via differential evolution
    2. Local refinement via L-BFGS-B
    
    Args:
        observations: dict with wavelengths, qsca, qabs arrays
        n_bounds: tuple (min, max) for real refractive index
        k_bounds: tuple (min, max) for imaginary part
        d_bounds: tuple (min, max) for diameter (nm)
        seed: Random seed for differential evolution
    
    Returns:
        recon_params: dict with recovered n_real, k_imag, diameter
        recon_spectra: dict with reconstructed qsca, qabs, qext, g arrays
    """
    wl = observations['wavelengths']
    qsca_obs = observations['qsca']
    qabs_obs = observations['qabs']
    
    def cost_function(params):
        """
        Least-squares cost function for Mie inversion.
        
        J(n, k, d) = Σ_λ [ (Qsca_obs(λ) - Qsca_model(λ))² / Qsca_obs(λ)²
                          + (Qabs_obs(λ) - Qabs_model(λ))² / Qabs_obs(λ)² ]
        """
        n_real, k_imag, diameter = params
        
        try:
            qsca_model, qabs_model, _, _ = forward_operator(
                n_real, k_imag, diameter, wl
            )
            
            # Relative residuals (weighted least squares)
            qsca_weight = np.maximum(np.abs(qsca_obs), 1e-10)
            qabs_weight = np.maximum(np.abs(qabs_obs), 1e-10)
            
            res_sca = ((qsca_obs - qsca_model) / qsca_weight) ** 2
            res_abs = ((qabs_obs - qabs_model) / qabs_weight) ** 2
            
            return np.sum(res_sca) + np.sum(res_abs)
        except Exception:
            return 1e10
    
    bounds = [n_bounds, k_bounds, d_bounds]
    
    # Stage 1: Global optimization (differential evolution)
    print("  [INV] Stage 1: Differential evolution global search...")
    result_global = differential_evolution(
        cost_function,
        bounds=bounds,
        seed=seed,
        maxiter=200,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )
    print(f"  [INV] Global result: n={result_global.x[0]:.4f}, "
          f"k={result_global.x[1]:.6f}, d={result_global.x[2]:.2f} nm, "
          f"cost={result_global.fun:.6e}")
    
    # Stage 2: Local refinement
    print("  [INV] Stage 2: L-BFGS-B local refinement...")
    result_local = minimize(
        cost_function,
        x0=result_global.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12},
    )
    
    n_rec, k_rec, d_rec = result_local.x
    print(f"  [INV] Final result: n={n_rec:.4f}, k={k_rec:.6f}, "
          f"d={d_rec:.2f} nm, cost={result_local.fun:.6e}")
    
    # Compute reconstructed spectra
    qsca_rec, qabs_rec, qext_rec, g_rec = forward_operator(
        n_rec, k_rec, d_rec, wl
    )
    
    recon_params = {
        'n_real': float(n_rec),
        'k_imag': float(k_rec),
        'diameter': float(d_rec),
    }
    
    recon_spectra = {
        'qsca': qsca_rec,
        'qabs': qabs_rec,
        'qext': qext_rec,
        'g': g_rec,
    }
    
    return recon_params, recon_spectra
