import numpy as np

from scipy import sparse

from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use('Agg')

def build_diffusion_matrix(kappa_2d, h):
    """
    Build sparse stiffness matrix for -div(kappa * grad(u)) = f
    on interior nodes with spacing h.
    Uses harmonic averaging of kappa at cell interfaces.
    """
    Ny, Nx = kappa_2d.shape
    N = Ny * Nx
    rows, cols, vals = [], [], []

    for iy in range(Ny):
        for ix in range(Nx):
            i = iy * Nx + ix
            diag_val = 0.0

            for diy, dix in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                niy, nix = iy + diy, ix + dix
                if 0 <= niy < Ny and 0 <= nix < Nx:
                    k_avg = 2.0 * kappa_2d[iy, ix] * kappa_2d[niy, nix] / (
                        kappa_2d[iy, ix] + kappa_2d[niy, nix] + 1e-30)
                    coeff = k_avg / (h * h)
                    j = niy * Nx + nix
                    rows.append(i)
                    cols.append(j)
                    vals.append(-coeff)
                    diag_val += coeff
                else:
                    diag_val += kappa_2d[iy, ix] / (h * h)

            rows.append(i)
            cols.append(i)
            vals.append(diag_val)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

def solve_pde(kappa_2d, source_2d, h):
    """Solve -div(kappa grad u) = f for u on interior nodes."""
    A = build_diffusion_matrix(kappa_2d, h)
    return spsolve(A, source_2d.ravel())

def make_basis_centers(n_per_dim):
    """Regular grid of Gaussian basis centers in [0,1]^2."""
    margin = 0.15
    cx = np.linspace(margin, 1.0 - margin, n_per_dim)
    cy = np.linspace(margin, 1.0 - margin, n_per_dim)
    CX, CY = np.meshgrid(cx, cy)
    return np.column_stack([CX.ravel(), CY.ravel()])

def build_observation_operator(obs_iy, obs_ix, Ny, Nx):
    """Sparse observation matrix that picks values at sensor locations."""
    n_obs = len(obs_iy)
    N = Ny * Nx
    rows = np.arange(n_obs)
    cols = obs_iy * Nx + obs_ix
    return sparse.csr_matrix((np.ones(n_obs), (rows, cols)), shape=(n_obs, N))

def make_sources(N_grid, h):
    """Create multiple source terms for multi-experiment inversion."""
    x = np.linspace(h, 1.0 - h, N_grid)
    y = np.linspace(h, 1.0 - h, N_grid)
    X, Y = np.meshgrid(x, y)
    return [
        10.0 * np.ones((N_grid, N_grid)),
        20.0 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * 0.15**2)),
        15.0 * np.exp(-((X - 0.3)**2 + (Y - 0.7)**2) / (2 * 0.12**2)),
        10.0 * np.sin(np.pi * X) * np.sin(np.pi * Y),
    ]

def load_and_preprocess_data(N_grid, n_basis_per_dim, n_obs_dim, snr_db, seed=42):
    """
    Load and preprocess data for Bayesian inversion.
    
    Creates:
    - Grid coordinates
    - True kappa field
    - Basis function centers
    - Sources
    - Observation operator
    - Synthetic noisy observations
    
    Returns:
        dict containing all preprocessed data
    """
    np.random.seed(seed)
    
    h = 1.0 / (N_grid + 1)
    x = np.linspace(h, 1.0 - h, N_grid)
    y = np.linspace(h, 1.0 - h, N_grid)
    X, Y = np.meshgrid(x, y)
    
    # True kappa: constant + Gaussian bumps
    kappa_bg = 1.0
    kappa_true = kappa_bg * np.ones((N_grid, N_grid))
    kappa_true += 2.0 * np.exp(-((X-0.3)**2 + (Y-0.3)**2) / (2*0.08**2))
    kappa_true += 1.5 * np.exp(-((X-0.7)**2 + (Y-0.7)**2) / (2*0.10**2))
    kappa_true -= 0.5 * np.exp(-((X-0.5)**2 + (Y-0.6)**2) / (2*0.06**2))
    kappa_true = np.maximum(kappa_true, 0.3)
    
    # Basis setup
    centers = make_basis_centers(n_basis_per_dim)
    n_basis = len(centers)
    sigma_basis = 0.12
    
    # Sources
    sources = make_sources(N_grid, h)
    n_sources = len(sources)
    
    # Sensors
    obs_ix_1d = np.linspace(1, N_grid-2, n_obs_dim, dtype=int)
    obs_iy_1d = np.linspace(1, N_grid-2, n_obs_dim, dtype=int)
    OIX, OIY = np.meshgrid(obs_ix_1d, obs_iy_1d)
    obs_ix, obs_iy = OIX.ravel(), OIY.ravel()
    n_obs = len(obs_ix)
    B = build_observation_operator(obs_iy, obs_ix, N_grid, N_grid)
    
    # Synthetic data with noise
    obs_data_list = []
    noise_std = 0.0
    for src in sources:
        u_true = solve_pde(kappa_true, src, h)
        u_obs = B @ u_true
        sig_pow = np.mean(u_obs**2)
        noise_std = np.sqrt(sig_pow / 10**(snr_db/10))
        obs_data_list.append(u_obs + noise_std * np.random.randn(n_obs))
    noise_var = noise_std**2
    
    print(f"Grid: {N_grid}x{N_grid}, h={h:.4f}")
    print(f"True kappa range: [{kappa_true.min():.3f}, {kappa_true.max():.3f}]")
    print(f"Basis functions: {n_basis}, sigma={sigma_basis}")
    print(f"Source experiments: {n_sources}")
    print(f"Sensors: {n_obs}")
    print(f"SNR={snr_db}dB, noise_std~{noise_std:.6f}")
    
    return {
        'N_grid': N_grid,
        'h': h,
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'kappa_bg': kappa_bg,
        'kappa_true': kappa_true,
        'centers': centers,
        'n_basis': n_basis,
        'sigma_basis': sigma_basis,
        'sources': sources,
        'n_sources': n_sources,
        'obs_ix': obs_ix,
        'obs_iy': obs_iy,
        'n_obs': n_obs,
        'B': B,
        'obs_data_list': obs_data_list,
        'noise_var': noise_var,
        'noise_std': noise_std,
        'snr_db': snr_db,
    }
