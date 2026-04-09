import numpy as np

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
