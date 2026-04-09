import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.sparse import csr_matrix, eye as speye, kron as spkron, vstack as spvstack, diags

from scipy.ndimage import gaussian_filter

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(nx, ny, domain_size, c0, n_transducers, ring_radius, noise_level, seed):
    """
    Generate the sound-speed phantom and transducer positions,
    then build the ray matrix and compute noisy travel-time measurements.
    
    Returns:
        dict containing:
            - c_gt: ground truth sound speed model (nx, ny)
            - transducers: transducer positions (n_transducers, 2)
            - G: ray-path kernel matrix (sparse)
            - dt_noisy: noisy travel-time perturbations
            - pairs: list of (transmitter, receiver) pairs
            - config: dictionary of configuration parameters
    """
    rng = np.random.default_rng(seed)
    
    # Generate sound speed phantom
    x = np.linspace(-domain_size / 2, domain_size / 2, nx)
    y = np.linspace(-domain_size / 2, domain_size / 2, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Start with water background
    c_gt = np.ones((nx, ny)) * c0

    # Breast outline (circle)
    r_breast = 0.035
    breast_mask = xx**2 + yy**2 < r_breast**2
    c_gt[breast_mask] = 1480  # Fat/tissue average

    # Fibroglandular tissue region (higher speed)
    r_fibro = 0.02
    fibro_mask = (xx - 0.005)**2 + (yy + 0.003)**2 < r_fibro**2
    c_gt[fibro_mask & breast_mask] = 1520

    # Tumour (higher sound speed)
    r_tumour = 0.006
    tumour_mask = (xx + 0.008)**2 + (yy - 0.01)**2 < r_tumour**2
    c_gt[tumour_mask & breast_mask] = 1560

    # Small cyst (lower sound speed)
    r_cyst = 0.004
    cyst_mask = (xx - 0.012)**2 + (yy + 0.008)**2 < r_cyst**2
    c_gt[cyst_mask & breast_mask] = 1450

    # Smooth transitions
    c_gt = gaussian_filter(c_gt, sigma=1.0)

    # Generate transducer positions on ring
    angles = np.linspace(0, 2 * np.pi, n_transducers, endpoint=False)
    x_trans = ring_radius * np.cos(angles)
    y_trans = ring_radius * np.sin(angles)
    transducers = np.column_stack([x_trans, y_trans])

    # Build ray matrix
    x_edges = np.linspace(-domain_size / 2, domain_size / 2, nx + 1)
    y_edges = np.linspace(-domain_size / 2, domain_size / 2, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    n_trans = len(transducers)
    n_cells = nx * ny

    rows, cols, vals = [], [], []
    pairs = []
    ray_idx = 0

    for i in range(n_trans):
        for j in range(n_trans):
            if i == j:
                continue
            x0, y0 = transducers[i]
            x1, y1 = transducers[j]

            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            n_samples = max(int(dist / (0.3 * min(dx, dy))), 300)

            t_param = np.linspace(0, 1, n_samples)
            x_ray = x0 + (x1 - x0) * t_param
            y_ray = y0 + (y1 - y0) * t_param
            ds = dist / n_samples

            # Find cell indices
            i_cells = np.clip(((x_ray - x_edges[0]) / dx).astype(int), 0, nx - 1)
            j_cells = np.clip(((y_ray - y_edges[0]) / dy).astype(int), 0, ny - 1)
            cell_ids = i_cells * ny + j_cells

            unique_cells, counts = np.unique(cell_ids, return_counts=True)
            for c, cnt in zip(unique_cells, counts):
                rows.append(ray_idx)
                cols.append(c)
                vals.append(cnt * ds)

            pairs.append((i, j))
            ray_idx += 1

    G = csr_matrix((vals, (rows, cols)), shape=(ray_idx, n_cells))

    # Compute travel times with noise
    s = 1.0 / c_gt.ravel()
    s0 = 1.0 / c0
    ds = s - s0
    dt_clean = G @ ds

    noise = noise_level * np.std(dt_clean) * rng.standard_normal(len(dt_clean))
    dt_noisy = dt_clean + noise

    config = {
        'nx': nx,
        'ny': ny,
        'domain_size': domain_size,
        'c0': c0,
        'n_transducers': n_transducers,
        'ring_radius': ring_radius,
        'noise_level': noise_level,
        'seed': seed
    }

    return {
        'c_gt': c_gt,
        'transducers': transducers,
        'G': G,
        'dt_noisy': dt_noisy,
        'dt_clean': dt_clean,
        'pairs': pairs,
        'config': config
    }
