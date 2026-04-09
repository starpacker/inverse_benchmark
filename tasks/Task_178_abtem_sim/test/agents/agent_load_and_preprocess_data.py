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

def make_si110_potential(nx, ny, pixel_size):
    """
    Create 2D projected potential for Si [110] zone axis.
    Returns V(x,y) in Volts.
    """
    a, b = 3.84, 5.43  # Si [110] unit cell (Å)
    positions = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.25), (0.75, 0.75)]

    Lx = nx * pixel_size
    Ly = ny * pixel_size
    x = np.arange(nx) * pixel_size
    y = np.arange(ny) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')

    V = np.zeros((ny, nx), dtype=np.float64)
    sigma_atom = 0.30  # Gaussian width (Å)
    V0 = 15.0          # peak potential (V)

    for ix in range(int(np.ceil(Lx / a)) + 1):
        for iy in range(int(np.ceil(Ly / b)) + 1):
            for fx, fy in positions:
                ax = ix * a + fx * a
                ay = iy * b + fy * b
                V += V0 * np.exp(-((X - ax)**2 + (Y - ay)**2) / (2 * sigma_atom**2))
    return V

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

def load_and_preprocess_data(nx, ny, pixel_size, voltage_kv, cs_mm, true_defocus, true_thickness, noise_level, seed):
    """
    Load and preprocess data for HRTEM simulation.
    
    Creates the Si [110] projected potential and generates ground truth and noisy images.
    
    Parameters:
    -----------
    nx, ny : int
        Image dimensions in pixels
    pixel_size : float
        Pixel size in Angstroms
    voltage_kv : float
        Accelerating voltage in kV
    cs_mm : float
        Spherical aberration coefficient in mm
    true_defocus : float
        True defocus value in nm (negative = underfocus)
    true_thickness : float
        True sample thickness in nm
    noise_level : float
        Noise level as fraction of mean intensity
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing:
        - 'V_pot': Projected potential array (Volts)
        - 'gt_img': Ground truth noiseless image
        - 'noisy_img': Noisy observation
        - 'pixel_size': Pixel size in Angstroms
        - 'voltage_kv': Accelerating voltage
        - 'cs_mm': Spherical aberration
        - 'true_defocus': True defocus (nm)
        - 'true_thickness': True thickness (nm)
        - 'wavelength': Electron wavelength (Angstroms)
        - 'sigma_e': Interaction parameter
    """
    # Compute physical parameters
    lam = relativistic_wavelength_A(voltage_kv)
    sig = interaction_param(voltage_kv)
    
    print(f"  λ = {lam:.5f} Å,  σ_e = {sig:.6f} rad/(V·Å)")
    
    # Build Si [110] projected potential
    print("\n[1/5] Building Si [110] projected potential ...")
    V_pot = make_si110_potential(nx, ny, pixel_size)
    print(f"      V range: [{V_pot.min():.2f}, {V_pot.max():.2f}] V")
    print(f"      Max phase: {sig * V_pot.max() * true_thickness * 10:.3f} rad")
    
    # Generate ground truth image (noiseless)
    print("[2/5] Simulating GT and noisy images ...")
    
    # Forward model for GT
    ny_pot, nx_pot = V_pot.shape
    t_A = true_thickness * 10.0  # nm → Å
    phase = sig * V_pot * t_A
    psi_exit = np.exp(1j * phase)
    K = freq_grid(nx_pot, ny_pot, pixel_size)
    H = ctf(K, lam, true_defocus, cs_mm)
    psi_img = np.fft.ifft2(np.fft.fft2(psi_exit) * H)
    gt_img = np.abs(psi_img)**2
    
    # Generate noisy image
    rng = np.random.default_rng(seed)
    noisy_img = gt_img.copy()
    if noise_level > 0:
        noisy_img = noisy_img + rng.normal(0, noise_level * noisy_img.mean(), noisy_img.shape)
        noisy_img = np.maximum(noisy_img, 0.0)
    
    contrast = (gt_img.max() - gt_img.min()) / gt_img.mean()
    print(f"      GT range: [{gt_img.min():.6f}, {gt_img.max():.6f}], contrast={contrast:.4f}")
    
    data_dict = {
        'V_pot': V_pot,
        'gt_img': gt_img,
        'noisy_img': noisy_img,
        'pixel_size': pixel_size,
        'voltage_kv': voltage_kv,
        'cs_mm': cs_mm,
        'true_defocus': true_defocus,
        'true_thickness': true_thickness,
        'wavelength': lam,
        'sigma_e': sig
    }
    
    return data_dict
