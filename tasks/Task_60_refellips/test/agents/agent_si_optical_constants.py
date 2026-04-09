import matplotlib

matplotlib.use('Agg')

def si_optical_constants(wavelength_nm, n_si_633, k_si_633):
    """Approximate Si optical constants (simple dispersion)."""
    lam_um = wavelength_nm / 1000.0
    n = n_si_633 + 0.8 * (0.633 / lam_um - 1)
    k = k_si_633 * (0.633 / lam_um) ** 2
    return n, k
