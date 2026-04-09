import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack

from scipy.ndimage import gaussian_filter

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(nx, ny, n_stations, n_rays, c0, noise_level, seed,
                              lat_min, lat_max, lon_min, lon_max):
    """
    Generate synthetic velocity model, stations, ray paths, and noisy travel-time data.
    
    Returns:
        dict containing:
            - dm_gt: ground truth velocity perturbation (nx, ny)
            - dm_gt_flat: flattened ground truth
            - stations: station coordinates (n_stations, 2)
            - pairs: list of (src, rcv) station index pairs
            - dt_noisy: noisy travel-time residuals
            - G: kernel matrix (sparse)
            - config: dict of configuration parameters
    """
    rng = np.random.default_rng(seed)
    
    # Generate velocity model
    dm_gt = np.zeros((nx, ny))
    
    # Smooth background
    raw = rng.standard_normal((nx, ny))
    dm_gt += 0.03 * gaussian_filter(raw, sigma=8)
    
    # Fast anomaly (craton)
    cx, cy = nx // 3, ny // 3
    for i in range(nx):
        for j in range(ny):
            r2 = ((i - cx) / 8)**2 + ((j - cy) / 6)**2
            if r2 < 1:
                dm_gt[i, j] += 0.08 * (1 - r2)
    
    # Slow anomaly (basin)
    cx2, cy2 = 2 * nx // 3, 2 * ny // 3
    for i in range(nx):
        for j in range(ny):
            r2 = ((i - cx2) / 6)**2 + ((j - cy2) / 8)**2
            if r2 < 1:
                dm_gt[i, j] -= 0.06 * (1 - r2)
    
    # Generate stations
    lats = lat_min + (lat_max - lat_min) * rng.random(n_stations)
    lons = lon_min + (lon_max - lon_min) * rng.random(n_stations)
    stations = np.column_stack([lats, lons])
    
    # Generate ray paths
    pairs = []
    for _ in range(n_rays):
        i, j = rng.choice(n_stations, 2, replace=False)
        pairs.append((i, j))
    
    # Build kernel matrix
    lat_edges = np.linspace(lat_min, lat_max, nx + 1)
    lon_edges = np.linspace(lon_min, lon_max, ny + 1)
    dlat = lat_edges[1] - lat_edges[0]
    dlon = lon_edges[1] - lon_edges[0]
    deg2km_lat = 111.0
    deg2km_lon = 111.0 * np.cos(np.radians(0.5 * (lat_min + lat_max)))
    
    n_cells = nx * ny
    rows, cols, vals = [], [], []
    
    for r, (si, sj) in enumerate(pairs):
        lat0, lon0 = stations[si]
        lat1, lon1 = stations[sj]
        
        n_samples = 500
        t = np.linspace(0, 1, n_samples)
        lat_ray = lat0 + (lat1 - lat0) * t
        lon_ray = lon0 + (lon1 - lon0) * t
        
        total_length_km = np.sqrt(
            ((lat1 - lat0) * deg2km_lat)**2 +
            ((lon1 - lon0) * deg2km_lon)**2
        )
        ds = total_length_km / n_samples
        
        cell_lengths = np.zeros(n_cells)
        i_cells = np.clip(((lat_ray - lat_min) / dlat).astype(int), 0, nx - 1)
        j_cells = np.clip(((lon_ray - lon_min) / dlon).astype(int), 0, ny - 1)
        
        for k in range(n_samples):
            cell_idx = i_cells[k] * ny + j_cells[k]
            cell_lengths[cell_idx] += ds
        
        nz = np.nonzero(cell_lengths)[0]
        for c in nz:
            rows.append(r)
            cols.append(c)
            vals.append(cell_lengths[c])
    
    G = csr_matrix((vals, (rows, cols)), shape=(n_rays, n_cells))
    
    # Forward model to get clean travel times
    dm_gt_flat = dm_gt.ravel()
    dt_clean = G @ (dm_gt_flat / c0)
    
    # Add noise
    noise = noise_level * np.std(dt_clean) * rng.standard_normal(len(dt_clean))
    dt_noisy = dt_clean + noise
    
    config = {
        'nx': nx,
        'ny': ny,
        'n_stations': n_stations,
        'n_rays': n_rays,
        'c0': c0,
        'noise_level': noise_level,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max
    }
    
    return {
        'dm_gt': dm_gt,
        'dm_gt_flat': dm_gt_flat,
        'stations': stations,
        'pairs': pairs,
        'dt_noisy': dt_noisy,
        'dt_clean': dt_clean,
        'G': G,
        'config': config
    }
