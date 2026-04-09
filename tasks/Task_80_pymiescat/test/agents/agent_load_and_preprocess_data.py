import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(wavelengths, gt_diameter, gt_n_real, gt_k_imag, noise_level, seed=42):
    """
    Generate synthetic Mie scattering observations from known parameters.
    
    Args:
        wavelengths: Array of wavelengths (nm)
        gt_diameter: Ground truth particle diameter (nm)
        gt_n_real: Ground truth real part of refractive index
        gt_k_imag: Ground truth imaginary part of refractive index
        noise_level: Relative noise level on measurements
        seed: Random seed for reproducibility
    
    Returns:
        observations: dict with noisy Qsca, Qabs spectra and wavelengths
        ground_truth: dict with true parameters and clean spectra
    """
    np.random.seed(seed)
    
    print(f"  [FORWARD] Computing Mie spectra for d={gt_diameter} nm, "
          f"m={gt_n_real}+{gt_k_imag}j")
    
    # Forward model: compute clean spectra
    qsca_clean, qabs_clean, qext_clean, g_clean = forward_operator(
        gt_n_real, gt_k_imag, gt_diameter, wavelengths
    )
    
    # Add relative noise
    qsca_noisy = qsca_clean * (1 + noise_level * np.random.randn(len(wavelengths)))
    qabs_noisy = qabs_clean * (1 + noise_level * np.random.randn(len(wavelengths)))
    qext_noisy = qsca_noisy + qabs_noisy  # Extinction = scattering + absorption
    
    observations = {
        'wavelengths': wavelengths,
        'qsca': qsca_noisy,
        'qabs': qabs_noisy,
        'qext': qext_noisy,
    }
    
    ground_truth = {
        'n_real': gt_n_real,
        'k_imag': gt_k_imag,
        'diameter': gt_diameter,
        'qsca_clean': qsca_clean,
        'qabs_clean': qabs_clean,
        'qext_clean': qext_clean,
        'g_clean': g_clean,
    }
    
    print(f"  [FORWARD] Qsca range: [{qsca_clean.min():.4f}, {qsca_clean.max():.4f}]")
    print(f"  [FORWARD] Qabs range: [{qabs_clean.min():.4f}, {qabs_clean.max():.4f}]")
    print(f"  [FORWARD] Qext range: [{qext_clean.min():.4f}, {qext_clean.max():.4f}]")
    
    return observations, ground_truth

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
