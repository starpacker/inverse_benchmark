import matplotlib

matplotlib.use('Agg')

SIGMA_RAY_REF = 5.31e-31

WAV_RAY_REF = 0.35e-6

def compute_rayleigh(wavelengths):
    """
    Rayleigh scattering cross-section for H₂ (λ⁻⁴ dependence).
    """
    return SIGMA_RAY_REF * (WAV_RAY_REF / wavelengths) ** 4
