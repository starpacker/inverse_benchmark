import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_sky_model(nx=128, ny=128, rng=None):
    """Create a synthetic sky model with point sources and extended emission."""
    if rng is None:
        rng = np.random.default_rng(42)

    sky = np.zeros((ny, nx), dtype=np.float64)

    # 3 point sources at different positions/fluxes
    point_sources = [
        (64, 64, 1.0),    # center, bright
        (40, 80, 0.5),    # offset, medium
        (90, 45, 0.3),    # offset, dim
    ]
    for y, x, flux in point_sources:
        sky[y, x] = flux

    # 1 extended Gaussian source (simulating a galaxy)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = 75, 55
    sigma_x, sigma_y = 5.0, 3.5
    angle = np.pi / 6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = xx - cx
    dy = yy - cy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    gauss = 0.4 * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    sky += gauss

    return sky

def generate_uv_coverage(n_antennas=10, n_hours=6, n_time_steps=60, rng=None):
    """
    Simulate (u,v) coverage from an interferometric array via
    earth-rotation synthesis.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    max_baseline = 50.0
    ant_E = rng.uniform(-max_baseline, max_baseline, n_antennas)
    ant_N = rng.uniform(-max_baseline, max_baseline, n_antennas)

    dec = np.deg2rad(45.0)
    ha = np.linspace(-n_hours / 2, n_hours / 2, n_time_steps) * (np.pi / 12)

    u_all, v_all = [], []
    for i in range(n_antennas):
        for j in range(i + 1, n_antennas):
            bE = ant_E[j] - ant_E[i]
            bN = ant_N[j] - ant_N[i]
            u_t = bE * np.cos(ha) - bN * np.sin(ha) * np.sin(dec)
            v_t = bE * np.sin(ha) * np.sin(dec) + bN * np.cos(ha) * np.cos(dec)
            u_all.append(u_t)
            v_all.append(v_t)

    u = np.concatenate(u_all)
    v = np.concatenate(v_all)

    # Include conjugate baselines (Hermitian symmetry)
    u = np.concatenate([u, -u])
    v = np.concatenate([v, -v])

    return u, v

def uv_to_grid_indices(u, v, nx, ny):
    """Convert continuous (u,v) to nearest grid indices for FFT grid."""
    ui = np.round(u).astype(int) % nx
    vi = np.round(v).astype(int) % ny
    return ui, vi

def make_dirty_image(vis, ui, vi, nx, ny):
    """Create the dirty image (adjoint applied to visibilities), normalized."""
    grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(grid, (vi, ui), vis)
    psf_grid = np.zeros((ny, nx), dtype=complex)
    np.add.at(psf_grid, (vi, ui), 1.0)
    dirty = np.fft.ifft2(grid).real
    psf = np.fft.ifft2(psf_grid).real
    peak_psf = psf.max()
    if peak_psf > 0:
        dirty /= peak_psf
    return dirty

def load_and_preprocess_data(nx=128, ny=128, n_antennas=10, n_hours=6, 
                              n_time_steps=60, noise_snr=30.0, seed=42):
    """
    Create synthetic sky model, generate UV coverage, compute visibilities,
    and add noise.
    
    Returns:
        dict containing:
            - sky_gt: ground truth sky image
            - vis_noisy: noisy visibilities
            - ui, vi: grid indices
            - u_unique, v_unique: unique UV coordinates
            - dirty: dirty image
            - nx, ny: image dimensions
    """
    rng = np.random.default_rng(seed)
    
    # Create sky model
    print("=" * 60)
    print("Step 1: Creating sky model ...")
    sky_gt = create_sky_model(nx, ny, rng=rng)
    print(f"  Sky shape: {sky_gt.shape}, max flux: {sky_gt.max():.4f}")
    
    # Generate UV coverage
    print("Step 2: Generating (u,v) coverage ...")
    u, v = generate_uv_coverage(n_antennas=n_antennas, n_hours=n_hours, 
                                 n_time_steps=n_time_steps, rng=rng)
    print(f"  Number of (u,v) points (incl. conjugates): {len(u)}")
    
    # Map to grid indices
    ui, vi = uv_to_grid_indices(u, v, nx, ny)
    
    # Remove duplicate grid points for cleaner sampling
    uv_pairs = np.stack([ui, vi], axis=1)
    _, unique_idx = np.unique(uv_pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    ui = ui[unique_idx]
    vi = vi[unique_idx]
    u_unique = u[unique_idx]
    v_unique = v[unique_idx]
    print(f"  Unique grid points: {len(ui)}")
    
    # Compute visibilities using forward operator
    print("Step 3: Computing visibilities ...")
    ft = np.fft.fft2(sky_gt)
    vis_true = ft[vi, ui]
    
    # Add noise
    signal_power = np.mean(np.abs(vis_true) ** 2)
    noise_std = np.sqrt(signal_power / noise_snr)
    noise = noise_std * (rng.standard_normal(len(vis_true)) +
                         1j * rng.standard_normal(len(vis_true))) / np.sqrt(2)
    vis_noisy = vis_true + noise
    actual_snr = np.sqrt(np.mean(np.abs(vis_true) ** 2) / np.mean(np.abs(noise) ** 2))
    print(f"  Noise std: {noise_std:.4e}, actual SNR: {actual_snr:.1f}")
    
    # Make dirty image
    print("Step 4: Making dirty image ...")
    dirty = make_dirty_image(vis_noisy, ui, vi, nx, ny)
    print(f"  Dirty image range: [{dirty.min():.4f}, {dirty.max():.4f}]")
    
    return {
        'sky_gt': sky_gt,
        'vis_noisy': vis_noisy,
        'ui': ui,
        'vi': vi,
        'u_unique': u_unique,
        'v_unique': v_unique,
        'dirty': dirty,
        'nx': nx,
        'ny': ny
    }
