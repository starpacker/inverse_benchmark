import numpy as np

def relativistic_wavelength_A(voltage_kv):
    """Relativistic de Broglie wavelength in Angstroms."""
    V = voltage_kv * 1e3  # Volts
    return 12.2643 / np.sqrt(V + 0.97845e-6 * V**2)
