import numpy as np

import matplotlib

matplotlib.use("Agg")

def _generate_particles(n, nx, ny, dx, zlo, zhi, rlo, rhi, margin, rng):
    """Generate n particles with random positions and radii."""
    pts = []
    sep_xy = 25 * dx
    sep_z = 40e-6
    for _ in range(n * 100):
        if len(pts) >= n:
            break
        x = rng.uniform(margin * dx, (nx - margin) * dx)
        y = rng.uniform(margin * dx, (ny - margin) * dx)
        z = rng.uniform(zlo, zhi)
        r = rng.uniform(rlo, rhi)
        if all(not (abs(x - p[0]) < sep_xy and abs(y - p[1]) < sep_xy and abs(z - p[2]) < sep_z) for p in pts):
            pts.append((x, y, z, r))
    return np.array(pts)

def load_and_preprocess_data(
    n_particles,
    nx,
    ny,
    pixel_size,
    z_min,
    z_max,
    r_min,
    r_max,
    margin,
    random_seed
):
    """
    Generate synthetic particle data and simulate inline hologram.
    
    Parameters:
    -----------
    n_particles : int
        Number of particles to generate
    nx, ny : int
        Image dimensions in pixels
    pixel_size : float
        Detector pixel pitch (m)
    z_min, z_max : float
        Z-range for particle positions (m)
    r_min, r_max : float
        Radius range for particles (m)
    margin : int
        Margin in pixels from edge
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    gt_particles : ndarray
        Ground truth particle array (N, 4) with (x, y, z, radius)
    gt_positions : ndarray
        Ground truth positions (N, 3) with (x, y, z)
    """
    rng = np.random.RandomState(random_seed)
    
    # Generate particles
    gt_particles = _generate_particles(
        n_particles, nx, ny, pixel_size,
        z_min, z_max, r_min, r_max, margin, rng
    )
    
    # Extract just positions (x, y, z)
    gt_positions = gt_particles[:, :3]
    
    print(f"  {len(gt_particles)} particles  z∈[{gt_positions[:, 2].min() * 1e6:.0f},{gt_positions[:, 2].max() * 1e6:.0f}] μm")
    
    return gt_particles, gt_positions
