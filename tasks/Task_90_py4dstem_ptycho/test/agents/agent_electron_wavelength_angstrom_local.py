import matplotlib

matplotlib.use("Agg")

def electron_wavelength_angstrom_local(E_eV):
    """Relativistic de Broglie wavelength [Å]."""
    import math as ma
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34
    lam = (h / ma.sqrt(2 * m * e * E_eV)
           / ma.sqrt(1 + e * E_eV / 2 / m / c**2) * 1e10)
    return lam
