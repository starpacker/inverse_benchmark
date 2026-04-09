import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(gt_params, param_order, freq, noise_level, seed=42):
    """
    Generate ground truth impedance spectrum and add noise.
    
    Parameters
    ----------
    gt_params : dict
        Ground truth circuit parameters.
    param_order : list
        Order of parameters in the vector.
    freq : ndarray
        Frequency array in Hz.
    noise_level : float
        Noise level as fraction of |Z|.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    data : dict
        Contains freq, Z_true, Z_noisy, gt_vec, param_order, noise_level.
    """
    np.random.seed(seed)
    
    gt_vec = np.array([gt_params[k] for k in param_order])
    
    # Compute ground truth impedance
    Z_true = forward_operator(gt_vec, freq)
    
    # Add noise (noise_level % of |Z| Gaussian noise to Re and Im)
    noise_re = noise_level * np.abs(Z_true) * np.random.randn(len(freq))
    noise_im = noise_level * np.abs(Z_true) * np.random.randn(len(freq))
    Z_noisy = Z_true + noise_re + 1j * noise_im
    
    data = {
        'freq': freq,
        'Z_true': Z_true,
        'Z_noisy': Z_noisy,
        'gt_vec': gt_vec,
        'param_order': param_order,
        'noise_level': noise_level,
    }
    
    return data

def forward_operator(params_vec, freq):
    """
    Compute complex impedance of a Randles circuit with Warburg element.
    
    Circuit: R0-p(R1,C1)-W (Randles circuit)
        R0 = series/ohmic resistance
        R1 = charge transfer resistance
        C1 = double-layer capacitance
        sigma_W = Warburg coefficient
    
    Z(ω) = R0 + Z_RC(ω) + Z_W(ω)
    where Z_RC = R1 / (1 + jωR1C1)   (parallel RC)
          Z_W  = σ_W / sqrt(ω) * (1 - j)  (Warburg impedance)
    
    Parameters
    ----------
    params_vec : ndarray or list
        [R0, R1, C1, sigma_W] circuit parameters.
    freq : ndarray
        Frequency array in Hz.
    
    Returns
    -------
    Z : ndarray (complex)
        Complex impedance at each frequency.
    """
    R0, R1, C1, sigma_W = params_vec
    omega = 2.0 * np.pi * freq
    
    # Parallel RC element: Z_RC = R1 / (1 + j*omega*R1*C1)
    Z_RC = R1 / (1.0 + 1j * omega * R1 * C1)
    
    # Warburg element: Z_W = sigma_W / sqrt(omega) * (1 - j)
    Z_W = sigma_W / np.sqrt(omega) * (1.0 - 1j)
    
    # Total impedance
    Z = R0 + Z_RC + Z_W
    
    return Z
