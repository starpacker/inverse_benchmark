import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def create_crescent_image(N, fov):
    """Create a crescent-shaped black hole shadow model."""
    x = np.linspace(-fov / 2, fov / 2, N)
    y = np.linspace(-fov / 2, fov / 2, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    r_ring = 40.0
    width = 12.0
    ring = np.exp(-0.5 * ((R - r_ring) / (width / 2.35)) ** 2)
    asym = 1.0 + 0.6 * np.cos(np.arctan2(Y, X) - np.pi)
    image = ring * asym
    shadow = 1.0 - np.exp(-0.5 * (R / (r_ring * 0.5)) ** 2)
    image *= shadow
    image = np.maximum(image, 0)
    if image.sum() > 0:
        image /= image.sum()
    return image

def generate_uv_coverage(n_stations, obs_hours, n_time):
    """Generate uv-coverage for an EHT-like VLBI array."""
    stations_lat = np.array([19.8, 37.1, -23.0, 32.7, -67.8, 78.2, 28.3, -30.7])[:n_stations]
    stations_lon = np.array([-155.5, -3.4, -67.8, -109.9, -68.8, 15.5, -16.6, 21.4])[:n_stations]
    lat_rad = np.deg2rad(stations_lat)
    lon_rad = np.deg2rad(stations_lon)
    wavelength_m = 1.3e-3
    earth_radius_m = 6.371e6
    R_lambda = earth_radius_m / wavelength_m
    X_st = R_lambda * np.cos(lat_rad) * np.cos(lon_rad)
    Y_st = R_lambda * np.cos(lat_rad) * np.sin(lon_rad)
    Z_st = R_lambda * np.sin(lat_rad)
    dec = np.deg2rad(12.0)
    ha = np.linspace(-obs_hours / 2, obs_hours / 2, n_time) * (np.pi / 12.0)
    u_all, v_all = [], []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            dx = X_st[j] - X_st[i]
            dy = Y_st[j] - Y_st[i]
            dz = Z_st[j] - Z_st[i]
            for h in ha:
                u = np.sin(h) * dx + np.cos(h) * dy
                v = (-np.sin(dec) * np.cos(h) * dx + np.sin(dec) * np.sin(h) * dy + np.cos(dec) * dz)
                u_all.append(u)
                v_all.append(v)
                u_all.append(-u)
                v_all.append(-v)
    u_all = np.array(u_all)
    v_all = np.array(v_all)
    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    u_all *= uas_to_rad
    v_all *= uas_to_rad
    return u_all, v_all

def load_and_preprocess_data(n_pix, fov_uas, n_stations, obs_hours, n_time, noise_level, seed):
    """
    Load and preprocess data for VLBI imaging.
    
    Creates ground truth image, generates UV coverage, and simulates noisy visibilities.
    
    Parameters
    ----------
    n_pix : int
        Image side length in pixels
    fov_uas : float
        Field of view in micro-arcseconds
    n_stations : int
        Number of VLBI stations
    obs_hours : float
        Observation duration in hours
    n_time : int
        Number of time samples
    noise_level : float
        Relative noise level for visibilities
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - gt_image: Ground truth black hole shadow image
        - u, v: UV coordinates
        - vis_noisy: Noisy visibility measurements
        - vis_true: True (noiseless) visibilities
        - fov_uas: Field of view
        - n_pix: Image size
    """
    np.random.seed(seed)
    
    # Create ground truth image
    gt_image = create_crescent_image(n_pix, fov_uas)
    
    # Generate UV coverage
    u, v = generate_uv_coverage(n_stations, obs_hours, n_time)
    
    # Forward observe to get visibilities
    N = gt_image.shape[0]
    pix_size = fov_uas / N
    x = (np.arange(N) - N / 2) * pix_size
    y = (np.arange(N) - N / 2) * pix_size
    X, Y = np.meshgrid(x, y)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    img_flat = gt_image.ravel()
    n_vis = len(u)
    vis_true = np.zeros(n_vis, dtype=complex)
    batch = 200
    for start in range(0, n_vis, batch):
        end = min(start + batch, n_vis)
        phase = -2.0 * np.pi * (np.outer(u[start:end], x_flat) + np.outer(v[start:end], y_flat))
        vis_true[start:end] = np.dot(np.exp(1j * phase), img_flat)
    
    noise_amp = noise_level * np.abs(vis_true).max()
    noise = noise_amp * (np.random.randn(n_vis) + 1j * np.random.randn(n_vis)) / np.sqrt(2)
    vis_noisy = vis_true + noise
    
    return {
        'gt_image': gt_image,
        'u': u,
        'v': v,
        'vis_noisy': vis_noisy,
        'vis_true': vis_true,
        'fov_uas': fov_uas,
        'n_pix': n_pix
    }
