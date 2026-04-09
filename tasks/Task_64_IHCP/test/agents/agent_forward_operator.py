import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import solve

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
