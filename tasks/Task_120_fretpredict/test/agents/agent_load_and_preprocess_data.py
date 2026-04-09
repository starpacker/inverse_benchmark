import numpy as np

import matplotlib

matplotlib.use("Agg")

def true_distance_pdf(r):
    """Mixture of two Gaussians: w1·N(mu1,sig1²) + w2·N(mu2,sig2²)."""
    w1, mu1, sig1 = 0.6, 4.0, 0.5
    w2, mu2, sig2 = 0.4, 7.0, 0.8
    g1 = w1 / (sig1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((r - mu1) / sig1) ** 2)
    g2 = w2 / (sig2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((r - mu2) / sig2) ** 2)
    return g1 + g2

def sample_distances(n, r_max):
    """Sample n distances from the Gaussian mixture."""
    n1 = int(0.6 * n)
    n2 = n - n1
    d1 = np.random.normal(4.0, 0.5, n1)
    d2 = np.random.normal(7.0, 0.8, n2)
    distances = np.concatenate([d1, d2])
    np.random.shuffle(distances)
    distances = np.clip(distances, 0.01, r_max)
    return distances

def load_and_preprocess_data(r_min, r_max, n_rbins, n_samples, R0, seed=42):
    """
    Generate ground truth distance distribution and sample FRET efficiencies.
    
    Returns:
        r_grid: distance grid array
        p_gt: ground truth distribution (normalized)
        distances: sampled distances
        params: dictionary of parameters
    """
    np.random.seed(seed)
    
    r_grid = np.linspace(r_min, r_max, n_rbins)
    dr = r_grid[1] - r_grid[0]
    
    p_gt = true_distance_pdf(r_grid)
    p_gt /= (np.sum(p_gt) * dr)
    
    distances = sample_distances(n_samples, r_max)
    
    params = {
        'r_min': r_min,
        'r_max': r_max,
        'n_rbins': n_rbins,
        'n_samples': n_samples,
        'R0': R0,
        'dr': dr,
        'seed': seed
    }
    
    return r_grid, p_gt, distances, params
