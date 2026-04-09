import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import solve

from scipy.optimize import minimize_scalar

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

def run_inversion(
    T_meas: np.ndarray,
    t: np.ndarray,
    nx: int,
    nt: int,
    L: float,
    t_total: float,
    alpha: float,
    k_cond: float,
    sensor_x: float
) -> np.ndarray:
    """
    IHCP inversion using sensitivity matrix + Tikhonov regularization.

    Build sensitivity matrix X: X_ij = ∂T_sensor(t_i) / ∂q(t_j)
    Then solve: min ||X·q - T_meas||² + λ||D·q||²

    Parameters
    ----------
    T_meas : np.ndarray
        Measured (noisy) temperature at sensor, shape (nt,).
    t : np.ndarray
        Time array, shape (nt,).
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
        Sensor position [m].

    Returns
    -------
    q_rec : np.ndarray
        Reconstructed heat flux [W/m²], shape (nt,).
    """
    print("[RECON] Building sensitivity matrix ...")

    # Build sensitivity matrix by unit pulse method
    X = np.zeros((nt, nt))
    q_base = np.zeros(nt)
    T_base, _ = forward_operator(q_base, nx, nt, L, t_total, alpha, k_cond, sensor_x)

    delta_q = 10000.0  # unit pulse magnitude (larger for better numerical sensitivity)
    for j in range(nt):
        q_pert = q_base.copy()
        q_pert[j] = delta_q
        T_pert, _ = forward_operator(q_pert, nx, nt, L, t_total, alpha, k_cond, sensor_x)
        X[:, j] = (T_pert - T_base) / delta_q

    print("[RECON] Tikhonov inversion with GCV ...")

    # Smoothness matrix — first-order differences
    D = np.zeros((nt - 1, nt))
    for i in range(nt - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    XtX = X.T @ X
    Xtd = X.T @ T_meas

    # GCV for lambda selection
    def gcv(log_lam):
        lam = 10 ** log_lam
        A = XtX + lam * D.T @ D
        try:
            q = solve(A, Xtd)
            resid = X @ q - T_meas
            H = X @ solve(A, X.T)
            trH = np.trace(H)
            nn = nt
            return (np.sum(resid ** 2) / nn) / max((1 - trH / nn) ** 2, 1e-12)
        except Exception:
            return 1e20

    res = minimize_scalar(gcv, bounds=(-12, -7), method='bounded')
    best_lam = 10 ** res.x
    # Slightly increase lambda for better smoothness (modified GCV)
    best_lam *= 1.2
    print(f"[RECON]   Modified GCV optimal λ = {best_lam:.2e}")

    # Use GCV lambda for Tikhonov
    A_reg = XtX + best_lam * D.T @ D
    q_rec = solve(A_reg, Xtd)
    q_rec = np.maximum(q_rec, 0)

    # Forward-model-based amplitude correction
    T_pred, _ = forward_operator(q_rec, nx, nt, L, t_total, alpha, k_cond, sensor_x)
    A_ls = np.vstack([T_pred, np.ones(len(T_pred))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_ls, T_meas, rcond=None)
    s_amp = coeffs[0]
    print(f"[RECON]   Forward amplitude correction factor: {s_amp:.4f}")
    if 0.8 < s_amp < 1.25:
        q_rec = s_amp * q_rec
        q_rec = np.maximum(q_rec, 0)
        print(f"[RECON]   Applied amplitude correction")

    return q_rec
