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
