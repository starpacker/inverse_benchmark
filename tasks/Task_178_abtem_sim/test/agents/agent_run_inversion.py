import numpy as np

from scipy.optimize import minimize

def relativistic_wavelength_A(voltage_kv):
    """Relativistic de Broglie wavelength in Angstroms."""
    V = voltage_kv * 1e3  # Volts
    return 12.2643 / np.sqrt(V + 0.97845e-6 * V**2)

def interaction_param(voltage_kv):
    """
    Interaction parameter sigma_e in rad/(V*Å).
    Formula: sigma = (2*pi / (lambda*V)) * (mc^2 + eV) / (2*mc^2 + eV)
    where lambda in Å, V in eV, mc^2 = 510998.95 eV.
    """
    lam = relativistic_wavelength_A(voltage_kv)  # Å
    V = voltage_kv * 1e3  # eV
    mc2 = 510998.95  # eV
    sigma = 2 * np.pi / (lam * V) * (mc2 + V) / (2 * mc2 + V)
    return sigma

def ctf(k, lam, defocus_nm, cs_mm):
    """
    Contrast transfer function H(k).
    defocus_nm: negative = underfocus (standard for HRTEM).
    """
    df_A = defocus_nm * 10.0
    Cs_A = cs_mm * 1e7

    chi = np.pi * lam * df_A * k**2 - 0.5 * np.pi * Cs_A * lam**3 * k**4

    # Spatial coherence envelope
    alpha = 0.5e-3
    E_s = np.exp(-0.5 * (np.pi * alpha)**2 * (df_A * k + Cs_A * lam**2 * k**3)**2)
    # Temporal coherence envelope
    delta_f = 30.0  # Å
    E_t = np.exp(-0.5 * (np.pi * lam * delta_f)**2 * k**4)

    return np.exp(-1j * chi) * E_s * E_t

def freq_grid(nx, ny, pixel_size):
    """2D spatial frequency magnitude |k| in 1/Å."""
    kx = np.fft.fftfreq(nx, d=pixel_size)
    ky = np.fft.fftfreq(ny, d=pixel_size)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    return np.sqrt(KX**2 + KY**2)

def forward_operator(V_pot, thickness_nm, defocus_nm, voltage_kv, cs_mm, pixel_size):
    """
    HRTEM forward model operator.
    
    Computes the HRTEM image from the projected potential using the phase-object
    approximation and contrast transfer function (CTF).
    
    Forward model:
      phase(x,y) = sigma_e * V(x,y) * t
      psi_exit   = exp(i * phase)
      Image      = |IFFT{ FFT{psi_exit} * CTF(k) }|^2
    
    Parameters:
    -----------
    V_pot : ndarray
        2D projected potential in Volts
    thickness_nm : float
        Sample thickness in nm
    defocus_nm : float
        Defocus value in nm (negative = underfocus)
    voltage_kv : float
        Accelerating voltage in kV
    cs_mm : float
        Spherical aberration coefficient in mm
    pixel_size : float
        Pixel size in Angstroms
        
    Returns:
    --------
    img : ndarray
        Simulated HRTEM image intensity
    """
    ny, nx = V_pot.shape
    lam = relativistic_wavelength_A(voltage_kv)
    sig = interaction_param(voltage_kv)
    t_A = thickness_nm * 10.0  # nm → Å

    # Phase-object approximation
    phase = sig * V_pot * t_A
    psi_exit = np.exp(1j * phase)

    # Apply CTF in Fourier space
    K = freq_grid(nx, ny, pixel_size)
    H = ctf(K, lam, defocus_nm, cs_mm)

    # Image formation
    psi_img = np.fft.ifft2(np.fft.fft2(psi_exit) * H)
    img = np.abs(psi_img)**2

    return img

def run_inversion(data_dict):
    """
    Run the inverse problem to estimate defocus and thickness.
    
    Uses a two-stage approach:
    1. Coarse grid search over defocus and thickness parameter space
    2. Local refinement using Nelder-Mead optimization
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing preprocessed data from load_and_preprocess_data:
        - 'V_pot': Projected potential
        - 'noisy_img': Noisy observation to fit
        - 'pixel_size', 'voltage_kv', 'cs_mm': Physical parameters
        
    Returns:
    --------
    result_dict : dict
        Dictionary containing:
        - 'estimated_defocus': Estimated defocus in nm
        - 'estimated_thickness': Estimated thickness in nm
        - 'recon_img': Reconstructed image using estimated parameters
        - 'final_mse': Final mean squared error
        - 'optimization_result': scipy optimization result object
    """
    V_pot = data_dict['V_pot']
    noisy_img = data_dict['noisy_img']
    pixel_size = data_dict['pixel_size']
    voltage_kv = data_dict['voltage_kv']
    cs_mm = data_dict['cs_mm']
    
    # Stage 1: Coarse grid search
    print("[3/5] Coarse grid search ...")
    df_grid = np.linspace(-100, 0, 41)
    t_grid = np.linspace(1, 10, 37)

    best_cost = np.inf
    best_df = df_grid[0]
    best_t = t_grid[0]
    
    for df in df_grid:
        for t in t_grid:
            sim = forward_operator(V_pot, t, df, voltage_kv, cs_mm, pixel_size)
            c = np.mean((sim - noisy_img)**2)
            if c < best_cost:
                best_cost = c
                best_df = df
                best_t = t

    print(f"      Best: df={best_df:.1f}, t={best_t:.2f}  (MSE={best_cost:.4e})")

    # Stage 2: Local refinement using Nelder-Mead
    print("[4/5] Refining ...")
    
    def cost_function(params):
        d, t = params
        if t < 0.1 or t > 20:
            return 1e10
        sim = forward_operator(V_pot, t, d, voltage_kv, cs_mm, pixel_size)
        return np.mean((sim - noisy_img)**2)

    res = minimize(
        cost_function, 
        [best_df, best_t], 
        method='Nelder-Mead',
        options={'xatol': 0.05, 'fatol': 1e-14, 'maxiter': 800, 'adaptive': True}
    )
    
    est_df, est_t = res.x
    print(f"      Refined: df={est_df:.3f}, t={est_t:.4f}")

    # Generate reconstruction with estimated parameters
    recon_img = forward_operator(V_pot, est_t, est_df, voltage_kv, cs_mm, pixel_size)

    result_dict = {
        'estimated_defocus': float(est_df),
        'estimated_thickness': float(est_t),
        'recon_img': recon_img,
        'final_mse': float(res.fun),
        'optimization_result': res
    }
    
    return result_dict
