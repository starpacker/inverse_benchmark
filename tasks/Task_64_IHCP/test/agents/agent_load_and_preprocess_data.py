import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import solve

def load_and_preprocess_data(
    nx: int,
    nt: int,
    L: float,
    t_total: float,
    alpha: float,
    k_cond: float,
    sensor_x: float,
    noise_level: float,
    seed: int
) -> dict:
    """
    Generate synthetic IHCP data: ground truth heat flux and noisy temperature measurements.

    Parameters
    ----------
    nx : int
        Number of spatial nodes.
    nt : int
        Number of time steps.
    L : float
        Slab thickness [m].
    t_total : float
        Total simulation time [s].
    alpha : float
        Thermal diffusivity [m²/s].
    k_cond : float
        Thermal conductivity [W/(m·K)].
    sensor_x : float
        Sensor position from heated surface [m].
    noise_level : float
        Standard deviation of temperature noise [°C].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 't': time array
        - 'q_gt': ground truth heat flux
        - 'T_sensor_clean': clean sensor temperature
        - 'T_sensor_noisy': noisy sensor temperature
        - 'T_field': full temperature field
        - 'params': dictionary of physical parameters
    """
    print("[DATA] Generating synthetic heat flux & temperature ...")

    t = np.linspace(0, t_total, nt)

    # Ground truth: pulsed heat flux with ramp and decay
    q_gt = np.zeros_like(t)
    # Ramp up
    mask1 = (t >= 1.0) & (t < 3.0)
    q_gt[mask1] = 5e4 * (t[mask1] - 1.0) / 2.0
    # Plateau
    mask2 = (t >= 3.0) & (t < 6.0)
    q_gt[mask2] = 5e4
    # Decay
    mask3 = (t >= 6.0) & (t < 8.0)
    q_gt[mask3] = 5e4 * (1 - (t[mask3] - 6.0) / 2.0)

    # Compute clean sensor temperature using forward operator
    T_sensor_clean, T_field = forward_operator(
        q_gt, nx, nt, L, t_total, alpha, k_cond, sensor_x
    )

    # Add noise
    rng = np.random.default_rng(seed)
    T_sensor_noisy = T_sensor_clean + noise_level * rng.standard_normal(nt)

    print(f"[DATA] q range: [{q_gt.min():.0f}, {q_gt.max():.0f}] W/m²")
    print(f"[DATA] T_sensor range: [{T_sensor_clean.min():.1f}, {T_sensor_clean.max():.1f}] °C")

    params = {
        'nx': nx,
        'nt': nt,
        'L': L,
        't_total': t_total,
        'alpha': alpha,
        'k_cond': k_cond,
        'sensor_x': sensor_x
    }

    data = {
        't': t,
        'q_gt': q_gt,
        'T_sensor_clean': T_sensor_clean,
        'T_sensor_noisy': T_sensor_noisy,
        'T_field': T_field,
        'params': params
    }

    return data

def forward_operator(
    q_flux: np.ndarray,
    nx: int,
    nt: int,
    L: float,
    t_total: float,
    alpha: float,
    k_cond: float,
    sensor_pos: float
) -> tuple:
    """
    Solve 1D heat equation with Crank-Nicolson and return temperature at sensor location.

    The heat equation: ∂T/∂t = α ∂²T/∂x²
    Boundary conditions: -k ∂T/∂x|₀ = q(t), T(L,t) = T_initial

    Parameters
    ----------
    q_flux : np.ndarray
        Heat flux at x=0 [W/m²], shape (nt,).
    nx : int
        Number of spatial nodes.
    nt : int
        Number of time steps.
    L : float
        Slab thickness [m].
    t_total : float
        Total simulation time [s].
    alpha : float
        Thermal diffusivity [m²/s].
    k_cond : float
        Thermal conductivity [W/(m·K)].
    sensor_pos : float
        Sensor position [m].

    Returns
    -------
    T_sensor : np.ndarray
        Temperature at sensor location [°C], shape (nt,).
    T_field : np.ndarray
        Full temperature field, shape (nx, nt).
    """
    dx = L / (nx - 1)
    dt = t_total / nt
    x = np.linspace(0, L, nx)

    r = alpha * dt / (2 * dx**2)  # Crank-Nicolson parameter

    # Initial condition
    T = np.zeros(nx)
    T_field = np.zeros((nx, nt))

    # Tridiagonal matrices for Crank-Nicolson
    # A * T^{n+1} = B * T^n + bc
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))

    for i in range(1, nx - 1):
        A[i, i - 1] = -r
        A[i, i] = 1 + 2 * r
        A[i, i + 1] = -r
        B[i, i - 1] = r
        B[i, i] = 1 - 2 * r
        B[i, i + 1] = r

    # Boundary conditions
    A[0, 0] = 1 + 2 * r
    A[0, 1] = -2 * r
    A[-1, -1] = 1
    B[0, 0] = 1 - 2 * r
    B[0, 1] = 2 * r
    B[-1, -1] = 1

    # Sensor index
    ix_sensor = int(np.argmin(np.abs(x - sensor_pos)))

    T_sensor = np.zeros(nt)

    for n in range(nt):
        rhs = B @ T
        # Neumann BC at x=0: -k dT/dx = q → dT/dx = -q/k
        rhs[0] += 2 * r * dx * q_flux[n] / k_cond
        T = solve(A, rhs)
        T_field[:, n] = T
        T_sensor[n] = T[ix_sensor]

    return T_sensor, T_field
