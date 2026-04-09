import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.special import hankel2

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_95_eispy2d"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def green2d(k0, r):
    """G(r) = (j/4) H_0^{(2)}(k0 r) – 2-D scalar Green's function."""
    r_safe = np.where(r < 1e-12, 1e-12, r)
    return (1j / 4.0) * hankel2(0, k0 * r_safe)

def load_and_preprocess_data(n_grid, domain_size, freq, c0, n_tx, n_rx, r_array, snr_db, seed=42):
    """
    Load and preprocess data for EM inverse scattering.
    
    Creates the phantom (ground truth dielectric contrast), array geometry,
    and generates noisy scattered field measurements.
    
    Parameters
    ----------
    n_grid : int
        Number of grid points per side
    domain_size : float
        Size of the investigation domain in meters
    freq : float
        Operating frequency in Hz
    c0 : float
        Speed of light
    n_tx : int
        Number of transmitters
    n_rx : int
        Number of receivers
    r_array : float
        Radius of the circular array
    snr_db : float
        Signal-to-noise ratio in dB
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - chi_gt: ground truth dielectric contrast (n_grid, n_grid)
        - gx, gy: grid coordinate vectors
        - tx_angles: transmitter angles
        - rx_pos: receiver positions (n_rx, 2)
        - y_noisy: noisy scattered field measurements
        - k0: wavenumber
        - ds: pixel area
        - snr_db: SNR in dB
    """
    # Create phantom - two dielectric cylinders
    x = np.linspace(-domain_size / 2, domain_size / 2, n_grid)
    y = np.linspace(-domain_size / 2, domain_size / 2, n_grid)
    X, Y = np.meshgrid(x, y)
    chi_gt = np.zeros((n_grid, n_grid), dtype=np.float64)
    
    # cylinder 1 – centre (+5 cm, +3 cm), radius 5 cm, chi=0.5
    mask1 = (X - 0.05) ** 2 + (Y - 0.03) ** 2 < 0.05 ** 2
    chi_gt[mask1] = 0.5
    
    # cylinder 2 – centre (-6 cm, -2 cm), radius 3 cm, chi=1.0
    mask2 = (X + 0.06) ** 2 + (Y + 0.02) ** 2 < 0.03 ** 2
    chi_gt[mask2] = 1.0
    
    gx, gy = x, y
    ds = (gx[1] - gx[0]) * (gy[1] - gy[0])
    
    # Array geometry
    tx_angles = np.linspace(0, 2 * np.pi, n_tx, endpoint=False)
    rx_angles = np.linspace(0, 2 * np.pi, n_rx, endpoint=False)
    rx_pos = np.column_stack([r_array * np.cos(rx_angles),
                              r_array * np.sin(rx_angles)])
    
    # Wavenumber
    k0 = 2.0 * np.pi * freq / c0
    
    # Build sensing matrix
    Xg, Yg = np.meshgrid(gx, gy)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    n_pix = pts.shape[0]
    
    A = np.zeros((n_tx * n_rx, n_pix), dtype=np.complex128)
    
    for l, theta in enumerate(tx_angles):
        d_hat = np.array([np.cos(theta), np.sin(theta)])
        E_inc = np.exp(1j * k0 * (pts @ d_hat))
        
        for m in range(n_rx):
            dr = np.sqrt((rx_pos[m, 0] - pts[:, 0]) ** 2
                         + (rx_pos[m, 1] - pts[:, 1]) ** 2)
            G = green2d(k0, dr)
            A[l * n_rx + m, :] = k0 ** 2 * G * E_inc * ds
    
    # Forward solve + noise
    chi_flat = chi_gt.ravel()
    y_clean = A @ chi_flat
    
    noise_power = np.linalg.norm(y_clean) / (10 ** (snr_db / 20))
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(y_clean.shape)
             + 1j * rng.standard_normal(y_clean.shape)) / np.sqrt(2)
    noise *= noise_power / np.linalg.norm(noise)
    y_noisy = y_clean + noise
    
    print(f"  Grid {n_grid}×{n_grid},  ds={ds:.4e} m²")
    print(f"  {n_tx} TX,  {n_rx} RX  on circle R={r_array*100:.0f} cm")
    print(f"  A shape = {A.shape}")
    print(f"  |y_clean| = {np.linalg.norm(y_clean):.4e}")
    print(f"  |noise|   = {np.linalg.norm(noise):.4e}")
    
    return {
        'chi_gt': chi_gt,
        'gx': gx,
        'gy': gy,
        'tx_angles': tx_angles,
        'rx_pos': rx_pos,
        'y_noisy': y_noisy,
        'k0': k0,
        'ds': ds,
        'snr_db': snr_db,
        'n_grid': n_grid
    }
