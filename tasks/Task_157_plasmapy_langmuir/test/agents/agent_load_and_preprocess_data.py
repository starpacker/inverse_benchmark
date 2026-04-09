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

def load_and_preprocess_data(test_cases, V_range=(-30, 30), n_points=500, noise_level=0.02, base_seed=42):
    """
    Load and preprocess data: synthesize noisy Langmuir probe I-V data from known parameters.
    
    Parameters
    ----------
    test_cases : list of dict
        Each dict contains T_e, n_e, V_p, I_ion_sat, label
    V_range : tuple
        Voltage range (min, max) in Volts
    n_points : int
        Number of voltage points
    noise_level : float
        Noise level as fraction of signal range
    base_seed : int
        Base random seed for reproducibility
    
    Returns
    -------
    preprocessed_data : list of dict
        Each dict contains:
        - 'V': voltage array
        - 'I_clean': clean current array
        - 'I_noisy': noisy current array
        - 'true_params': dict of true parameters
        - 'label': case label
    """
    preprocessed_data = []
    
    for i, tc in enumerate(test_cases):
        rng = np.random.default_rng(base_seed + i)
        V = np.linspace(V_range[0], V_range[1], n_points)
        
        # Compute clean I-V curve using forward model parameters
        T_e = tc["T_e"]
        n_e = tc["n_e"]
        V_p = tc["V_p"]
        I_ion_sat = tc["I_ion_sat"]
        
        T_e_K = T_e * EV_TO_K
        I_e_sat = electron_saturation_current(T_e, n_e)
        
        # Compute exponent with clipping to avoid overflow
        exponent = E_CHARGE * (V - V_p) / (K_BOLTZMANN * T_e_K)
        exponent = np.clip(exponent, -500, 500)
        
        I_clean = np.where(
            V < V_p,
            I_ion_sat + I_e_sat * np.exp(exponent),
            I_ion_sat + I_e_sat,
        )
        
        # Add noise
        noise_amplitude = noise_level * (np.max(I_clean) - np.min(I_clean))
        noise = rng.normal(0, noise_amplitude, size=I_clean.shape)
        I_noisy = I_clean + noise
        
        # Compute true floating potential
        V_f_true = floating_potential(T_e, n_e, V_p, I_ion_sat)
        
        preprocessed_data.append({
            'V': V,
            'I_clean': I_clean,
            'I_noisy': I_noisy,
            'true_params': {
                'T_e': T_e,
                'n_e': n_e,
                'V_p': V_p,
                'I_ion_sat': I_ion_sat,
                'V_f': V_f_true,
            },
            'label': tc["label"],
        })
    
    return preprocessed_data
