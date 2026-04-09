import numpy as np

import matplotlib

matplotlib.use('Agg')

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
