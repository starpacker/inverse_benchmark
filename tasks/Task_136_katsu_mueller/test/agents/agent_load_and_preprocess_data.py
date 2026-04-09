import matplotlib

matplotlib.use('Agg')

import numpy as np

from numpy.linalg import lstsq, norm

def mueller_rotation(theta):
    """Rotation matrix R(theta) for Mueller calculus (angle in radians)."""
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    return np.array([
        [1,  0,   0,   0],
        [0,  c2,  s2,  0],
        [0, -s2,  c2,  0],
        [0,  0,   0,   1],
    ])

def mueller_linear_retarder(delta, theta=0.0):
    """Mueller matrix of a linear retarder with retardance *delta* at angle *theta*."""
    cd = np.cos(delta)
    sd = np.sin(delta)
    M_ret = np.array([
        [1, 0,   0,    0],
        [0, 1,   0,    0],
        [0, 0,   cd,   sd],
        [0, 0,  -sd,   cd],
    ])
    R = mueller_rotation(theta)
    Rinv = mueller_rotation(-theta)
    return R @ M_ret @ Rinv

def mueller_ideal_polarizer_h():
    """Ideal horizontal linear polarizer."""
    return 0.5 * np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=float)

def mueller_partial_polarizer(diattenuation=0.5, theta=0.0):
    """Partial polarizer with specified diattenuation D ∈ [0, 1]."""
    D = diattenuation
    q = 0.5 * (1 + D)
    r = 0.5 * (1 - D)
    a = q + r
    b = q - r
    c = 2 * np.sqrt(q * r)
    M = np.array([
        [a, b, 0, 0],
        [b, a, 0, 0],
        [0, 0, c, 0],
        [0, 0, 0, c],
    ])
    if theta != 0.0:
        R = mueller_rotation(theta)
        Rinv = mueller_rotation(-theta)
        M = R @ M @ Rinv
    return M

def build_measurement_matrix(theta_g_list, theta_a_list, delta_g=np.pi/2, delta_a=np.pi/2):
    """
    Build the N×16 measurement (instrument) matrix **W** for a DRR polarimeter.

    Parameters
    ----------
    theta_g_list : array-like  – generator retarder angles (rad)
    theta_a_list : array-like  – analyser retarder angles (rad)
    delta_g      : float       – generator retarder retardance (default QWP)
    delta_a      : float       – analyser retarder retardance (default QWP)

    Returns
    -------
    W : ndarray, shape (N, 16)
    """
    N = len(theta_g_list)
    W = np.zeros((N, 16))

    P_h = mueller_ideal_polarizer_h()
    S0 = np.array([1.0, 0.0, 0.0, 0.0])

    for k in range(N):
        M_g = mueller_linear_retarder(delta_g, theta_g_list[k])
        M_psg = M_g @ P_h
        S_in = M_psg @ S0

        M_a = mueller_linear_retarder(delta_a, theta_a_list[k])
        M_psa = P_h @ M_a
        D_out = M_psa[0, :]

        W[k, :] = np.outer(D_out, S_in).ravel()

    return W

def load_and_preprocess_data(n_measurements=36, noise_sigma_factor=0.005, rng_seed=42):
    """
    Generate ground truth Mueller matrix and simulate noisy DRR measurements.
    
    Parameters
    ----------
    n_measurements : int
        Number of DRR measurement positions.
    noise_sigma_factor : float
        Noise level as fraction of max clean intensity.
    rng_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    data : dict
        Contains:
        - 'M_true': Ground truth Mueller matrix (4x4)
        - 'I_noisy': Noisy intensity measurements (N,)
        - 'I_clean': Clean intensity measurements (N,)
        - 'W': Measurement matrix (N, 16)
        - 'theta_g': Generator angles (N,)
        - 'theta_a': Analyzer angles (N,)
        - 'noise_sigma': Actual noise standard deviation
        - 'snr_db': Signal-to-noise ratio in dB
    """
    rng = np.random.default_rng(rng_seed)
    
    # Create ground truth Mueller matrix
    # Combination: partial polarizer (D=0.6, 30°) followed by
    # quarter-wave retarder at 15°
    M_pol = mueller_partial_polarizer(diattenuation=0.6, theta=np.deg2rad(30))
    M_ret = mueller_linear_retarder(delta=np.pi / 2, theta=np.deg2rad(15))
    M_true = M_ret @ M_pol
    # Normalize so M[0,0] = 1 (conventional)
    M_true = M_true / M_true[0, 0]
    
    # DRR measurement angles (5:1 ratio for generator:analyzer rotation)
    theta_g = np.linspace(0, np.pi, n_measurements, endpoint=False)
    theta_a = np.linspace(0, 5 * np.pi, n_measurements, endpoint=False)
    
    # Build measurement matrix
    W = build_measurement_matrix(theta_g, theta_a)
    
    # Simulate clean measurements
    m_vec = M_true.ravel()
    I_clean = W @ m_vec
    
    # Add Gaussian noise
    noise_sigma = noise_sigma_factor * np.max(np.abs(I_clean))
    noise = rng.normal(0, noise_sigma, size=I_clean.shape)
    I_noisy = I_clean + noise
    
    # Calculate SNR
    snr_db = 20 * np.log10(norm(I_clean) / norm(noise)) if norm(noise) > 0 else 100.0
    
    data = {
        'M_true': M_true,
        'I_noisy': I_noisy,
        'I_clean': I_clean,
        'W': W,
        'theta_g': theta_g,
        'theta_a': theta_a,
        'noise_sigma': noise_sigma,
        'snr_db': snr_db,
    }
    
    return data
