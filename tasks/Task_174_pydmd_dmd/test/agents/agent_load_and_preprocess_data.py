import numpy as np

import matplotlib

matplotlib.use("Agg")

def load_and_preprocess_data(nx, ny, nt, dt, snr_db, seed):
    """
    Synthesize a rank-4 spatiotemporal field from 2 oscillatory modes.
    
    Each oscillatory mode (complex eigenvalue λ = σ + jω) produces a
    conjugate pair of discrete eigenvalues μ, μ*.
    
    Parameters
    ----------
    nx, ny : int
        Spatial grid dimensions
    nt : int
        Number of time snapshots
    dt : float
        Time step (s)
    snr_db : float
        Signal-to-noise ratio for additive Gaussian noise
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    gt_field : ndarray (nx*ny, nt)
        Ground-truth snapshot matrix
    noisy_field : ndarray (nx*ny, nt)
        Noisy observations
    meta : dict
        Contains spatial coords, time, true eigenvalues, etc.
    """
    rng = np.random.default_rng(seed)
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    k = np.arange(nt)
    t = k * dt
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    
    # 4 spatial patterns (2 pairs: cos/sin parts)
    phi1 = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)
    phi2 = np.sin(2 * np.pi * Xg) * np.cos(np.pi * Yg)
    phi3 = np.cos(np.pi * Xg) * np.sin(2 * np.pi * Yg)
    phi4 = np.cos(2 * np.pi * Xg) * np.cos(2 * np.pi * Yg)
    
    # Continuous-time eigenvalues
    omega1, sigma1 = 2.3, -0.1
    omega2, sigma2 = 5.7, -0.05
    lam1 = sigma1 + 1j * omega1
    lam2 = sigma2 + 1j * omega2
    
    # Discrete-time eigenvalues
    mu1 = np.exp(lam1 * dt)
    mu2 = np.exp(lam2 * dt)
    
    # Temporal dynamics (real & imaginary parts from each conjugate pair)
    c1_re = np.real(mu1 ** k)
    c1_im = np.imag(mu1 ** k)
    c2_re = np.real(mu2 ** k)
    c2_im = np.imag(mu2 ** k)
    
    # Snapshot matrix: X = Σ φ_i ⊗ c_i (rank 4)
    gt_field = (np.outer(phi1.ravel(), c1_re)
                + np.outer(phi2.ravel(), c1_im)
                + np.outer(phi3.ravel(), c2_re)
                + np.outer(phi4.ravel(), c2_im))
    
    # Additive Gaussian noise
    signal_power = np.mean(gt_field ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), gt_field.shape)
    noisy_field = gt_field + noise
    
    meta = dict(
        x=x,
        y=y,
        t=t,
        dt=dt,
        nx=nx,
        ny=ny,
        nt=nt,
        true_continuous_eigenvalues=np.array([lam1, lam2]),
        true_discrete_eigenvalues=np.array([mu1, mu2]),
        omega=[omega1, omega2],
        sigma=[sigma1, sigma2],
    )
    
    return gt_field, noisy_field, meta
