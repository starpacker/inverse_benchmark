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

def floating_potential(T_e_eV, n_e, V_p, I_ion_sat):
    """Compute floating potential V_f where I(V_f) = 0."""
    I_e_sat = electron_saturation_current(T_e_eV, n_e)
    if I_e_sat <= 0 or -I_ion_sat <= 0:
        return V_p  # degenerate
    T_e_K = T_e_eV * EV_TO_K
    V_f = V_p + (K_BOLTZMANN * T_e_K / E_CHARGE) * np.log(-I_ion_sat / I_e_sat)
    return V_f
