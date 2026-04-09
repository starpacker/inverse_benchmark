import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.signal import sawtooth

def load_and_preprocess_data(sr, duration, noise_std, random_seed):
    """
    Synthesize source signals and create mixed observations.
    
    Parameters
    ----------
    sr : int
        Sample rate in Hz.
    duration : float
        Duration in seconds.
    noise_std : float
        Standard deviation of additive noise.
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    sources : np.ndarray
        Ground truth source signals, shape (2, N).
    mixed : np.ndarray
        Mixed observations, shape (n_sensors, N).
    mixing_matrix : np.ndarray
        The mixing matrix A, shape (n_sensors, 2).
    t : np.ndarray
        Time array, shape (N,).
    params : dict
        Dictionary with sample rate, duration, etc.
    """
    np.random.seed(random_seed)
    
    N = int(sr * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Source 1: two sinusoids (simulated "speaker 1")
    s1 = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)
    
    # Source 2: sinusoids + sawtooth (simulated "speaker 2")
    s2 = (0.4 * np.sin(2 * np.pi * 330 * t)
          + 0.3 * np.sin(2 * np.pi * 660 * t)
          + 0.3 * sawtooth(2 * np.pi * 110 * t))
    
    # Stack into (2, N)
    sources = np.vstack([s1, s2])
    
    # Mixing matrix — 5 sensors, 2 sources (overdetermined)
    A = np.array([[0.8, 0.4],
                  [0.3, 0.9],
                  [0.6, 0.5],
                  [0.9, 0.2],
                  [0.2, 0.8]])
    
    # Forward operator: y = A @ s + noise
    mixed = A @ sources + noise_std * np.random.randn(A.shape[0], N)
    
    params = {
        'sr': sr,
        'duration': duration,
        'noise_std': noise_std,
        'n_samples': N,
        'n_sources': 2,
        'n_sensors': A.shape[0],
    }
    
    print(f"[INFO] Source shape : {sources.shape}")
    print(f"[INFO] Mixed  shape : {mixed.shape}")
    print(f"[INFO] Mixing matrix A:\n{A}")
    
    return sources, mixed, A, t, params
