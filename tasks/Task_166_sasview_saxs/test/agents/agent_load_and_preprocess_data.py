import matplotlib

matplotlib.use('Agg')

import numpy as np

def sphere_form_factor_amplitude(q, R):
    """
    Normalised form-factor amplitude for a homogeneous sphere:
        f(q,R) = 3 [sin(qR) - qR cos(qR)] / (qR)^3
    Returns f(q,R).  P(q) = f^2.
    """
    qR = np.asarray(q * R, dtype=np.float64)
    result = np.ones_like(qR)
    mask = qR > 1e-12
    result[mask] = 3.0 * (np.sin(qR[mask]) - qR[mask] * np.cos(qR[mask])) / qR[mask]**3
    return result

def sphere_intensity(q, R, scale, background):
    """
    I(q) = scale * V * delta_rho^2 * P(q) + background
    where P(q) = |f(q,R)|^2 and V = (4/3)pi R^3.
    
    For simplicity we fold V*delta_rho^2 into the scale factor
    so:  I(q) = scale * P(q) + background
    This is the standard parameterisation used by SasView.
    """
    P_q = sphere_form_factor_amplitude(q, R)**2
    return scale * P_q + background

def load_and_preprocess_data(gt_radius, gt_scale, gt_background, 
                              q_min=0.001, q_max=0.5, n_points=200,
                              noise_level=0.02, seed=42):
    """
    Load and preprocess data: Generate synthetic SAXS data with noise.
    
    Parameters:
    -----------
    gt_radius : float
        Ground truth sphere radius in Angstroms
    gt_scale : float
        Ground truth scale factor
    gt_background : float
        Ground truth background level
    q_min : float
        Minimum q value (Å^-1)
    q_max : float
        Maximum q value (Å^-1)
    n_points : int
        Number of q points
    noise_level : float
        Relative noise level
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict containing:
        - q: scattering vector array
        - I_clean: clean intensity (ground truth)
        - I_noisy: noisy intensity (measured)
        - sigma: noise standard deviation array
        - gt_params: dictionary of ground truth parameters
    """
    print("[SYNTH] Generating synthetic SAXS data ...")
    
    q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
    I_clean = sphere_intensity(q, gt_radius, gt_scale, gt_background)
    
    np.random.seed(seed)
    sigma = noise_level * np.sqrt(np.abs(I_clean)) + 1e-6
    I_noisy = I_clean + np.random.normal(0, sigma)
    I_noisy = np.maximum(I_noisy, 1e-8)
    
    print(f"[SYNTH] q range: {q.min():.4f} – {q.max():.4f} Å^-1  ({len(q)} points)")
    print(f"[SYNTH] I(q) range: {I_clean.min():.6f} – {I_clean.max():.6f}")
    print(f"[SYNTH] GT params: R={gt_radius} Å, scale={gt_scale}, bg={gt_background}")
    
    data = {
        'q': q,
        'I_clean': I_clean,
        'I_noisy': I_noisy,
        'sigma': sigma,
        'gt_params': {
            'radius': gt_radius,
            'scale': gt_scale,
            'background': gt_background
        }
    }
    
    return data
