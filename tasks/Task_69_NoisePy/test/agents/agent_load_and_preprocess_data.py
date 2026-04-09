import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack as spvstack, diags

def load_and_preprocess_data(nx, ny, n_stations, seed, xmin, xmax, ymin, ymax, noise_level, c0):
    """
    Generate synthetic velocity model, station positions, and build kernel matrix.
    
    Returns:
        dict containing all necessary data for forward modeling and inversion
    """
    rng = np.random.default_rng(seed)
    
    # Generate velocity model: checkerboard + Gaussian anomalies
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Checkerboard pattern
    checker_scale = (xmax - xmin) / 5
    dm_gt = 0.05 * np.sin(2 * np.pi * xx / checker_scale) * \
            np.sin(2 * np.pi * yy / checker_scale)
    
    # Gaussian anomaly (high velocity)
    cx, cy = 0.6 * xmax, 0.4 * ymax
    sigma = 20.0
    dm_gt += 0.08 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    
    # Low-velocity zone
    cx2, cy2 = 0.3 * xmax, 0.7 * ymax
    dm_gt -= 0.06 * np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / (2 * (25)**2))
    
    # Generate random station positions
    x_sta = xmin + (xmax - xmin) * rng.random(n_stations)
    y_sta = ymin + (ymax - ymin) * rng.random(n_stations)
    stations = np.column_stack([x_sta, y_sta])
    
    # Build kernel matrix for all station pairs
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    n_cells = nx * ny
    
    pairs = []
    rows, cols, vals = [], [], []
    ray_idx = 0
    
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            x0, y0 = stations[i]
            x1, y1 = stations[j]
            
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            if dist < 10.0:  # Skip very short paths
                continue
            
            # Sample ray path
            n_samples = max(int(dist / (0.5 * min(dx, dy))), 200)
            t_param = np.linspace(0, 1, n_samples)
            x_ray = x0 + (x1 - x0) * t_param
            y_ray = y0 + (y1 - y0) * t_param
            ds = dist / n_samples
            
            # Accumulate path lengths per cell
            i_cells = np.clip(((x_ray - xmin) / dx).astype(int), 0, nx - 1)
            j_cells = np.clip(((y_ray - ymin) / dy).astype(int), 0, ny - 1)
            cell_ids = i_cells * ny + j_cells
            
            # Count segments per cell
            unique_cells, counts = np.unique(cell_ids, return_counts=True)
            for c, cnt in zip(unique_cells, counts):
                rows.append(ray_idx)
                cols.append(c)
                vals.append(cnt * ds)
            
            pairs.append((i, j))
            ray_idx += 1
    
    G = csr_matrix((vals, (rows, cols)), shape=(ray_idx, n_cells))
    
    return {
        'dm_gt': dm_gt,
        'dm_gt_flat': dm_gt.ravel(),
        'xx': xx,
        'yy': yy,
        'stations': stations,
        'G': G,
        'pairs': pairs,
        'nx': nx,
        'ny': ny,
        'c0': c0,
        'noise_level': noise_level,
        'rng': rng,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax
    }
