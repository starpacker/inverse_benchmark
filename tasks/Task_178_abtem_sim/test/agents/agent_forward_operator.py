import numpy as np

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
