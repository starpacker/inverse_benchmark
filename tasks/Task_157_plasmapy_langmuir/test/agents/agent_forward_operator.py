import numpy as np

import matplotlib

matplotlib.use("Agg")

E_CHARGE = 1.602176634e-19

M_ELECTRON = 9.1093837015e-31

K_BOLTZMANN = 1.380649e-23

EV_TO_K = E_CHARGE / K_BOLTZMANN

A_PROBE = 1.0e-6

def electron_saturation_current(T_e_eV, n_e):
    """Electron saturation current [A] for given T_e (eV) and n_e (m⁻³)."""
    T_e_K = T_e_eV * EV_TO_K
    return n_e * E_CHARGE * A_PROBE * np.sqrt(K_BOLTZMANN * T_e_K / (2 * np.pi * M_ELECTRON))

def forward_operator(V, T_e, n_e, V_p, I_ion_sat):
    """
    Theoretical Langmuir probe I-V characteristic (forward model).
    
    Parameters
    ----------
    V : array_like
        Bias voltage [V]
    T_e : float
        Electron temperature [eV]
    n_e : float
        Electron density [m⁻³]
    V_p : float
        Plasma potential [V]
    I_ion_sat : float
        Ion saturation current [A] (negative)
    
    Returns
    -------
    I : ndarray
        Probe current [A]
    """
    T_e_K = T_e * EV_TO_K
    I_e_sat = electron_saturation_current(T_e, n_e)
    
    # Clamp the exponent to avoid overflow
    exponent = E_CHARGE * (V - V_p) / (K_BOLTZMANN * T_e_K)
    exponent = np.clip(exponent, -500, 500)
    
    I = np.where(
        V < V_p,
        I_ion_sat + I_e_sat * np.exp(exponent),
        I_ion_sat + I_e_sat,
    )
    return I
