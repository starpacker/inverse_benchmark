import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

GRAV_CONST = 6.674e-3

def prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho):
    """
    Compute vertical gravity component gz for a rectangular prism.
    
    Uses the analytical formula from Blakely (1996) / Nagy (1966).
    
    Parameters
    ----------
    x1, x2 : float  Prism x bounds
    y1, y2 : float  Prism y bounds
    z1, z2 : float  Prism z bounds (z positive downward in convention, but we use z negative for depth)
    xp, yp, zp : float  Observation point coordinates
    rho : float  Density contrast [g/cm³]
    
    Returns
    -------
    gz : float  Vertical gravity component [mGal]
    """
    # Shift coordinates relative to observation point
    dx = [x1 - xp, x2 - xp]
    dy = [y1 - yp, y2 - yp]
    dz = [z1 - zp, z2 - zp]
    
    gz = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = dx[i]
                y = dy[j]
                z = dz[k]
                r = np.sqrt(x**2 + y**2 + z**2)
                
                # Avoid singularities
                r = max(r, 1e-10)
                
                # Sign for the sum
                sign = (-1) ** (i + j + k)
                
                # Compute terms
                term1 = 0.0
                term2 = 0.0
                term3 = 0.0
                
                # x * ln(y + r)
                if abs(y + r) > 1e-10:
                    term1 = x * np.log(y + r)
                
                # y * ln(x + r)
                if abs(x + r) > 1e-10:
                    term2 = y * np.log(x + r)
                
                # z * arctan(xy / (zr))
                denom = z * r
                if abs(denom) > 1e-10:
                    term3 = z * np.arctan2(x * y, denom)
                
                gz += sign * (term1 + term2 - term3)
    
    # Convert to mGal (G in appropriate units)
    # G = 6.674e-11 m³/(kg·s²), density in g/cm³ = 1000 kg/m³
    # 1 mGal = 1e-5 m/s²
    # gz = G * rho * integral, with rho in g/cm³
    # Factor: 6.674e-11 * 1000 * 1e5 = 6.674e-3
    gz *= GRAV_CONST * rho
    
    return gz

def build_sensitivity_matrix(mesh_info, rx_locs):
    """
    Build the sensitivity (Jacobian) matrix G for the linear inverse problem.
    
    G[i, j] = dg_z(rx_i) / d_rho(cell_j)
    
    For gravity, this is the Green's function contribution from each cell.
    """
    n_rx = rx_locs.shape[0]
    n_cells = mesh_info['n_cells']
    cc = mesh_info['cell_centers']
    
    dx = mesh_info['hx'][0] / 2
    dy = mesh_info['hy'][0] / 2
    dz = mesh_info['hz'][0] / 2
    
    G = np.zeros((n_rx, n_cells))
    
    for i_rx in range(n_rx):
        xp, yp, zp = rx_locs[i_rx]
        
        for i_cell in range(n_cells):
            xc, yc, zc = cc[i_cell]
            
            # Prism bounds
            x1, x2 = xc - dx, xc + dx
            y1, y2 = yc - dy, yc + dy
            z1, z2 = zc - dz, zc + dz
            
            # Sensitivity = gz for unit density
            G[i_rx, i_cell] = prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, 1.0)
    
    return G

def run_inversion(data_dict):
    """
    Tikhonov-regularized gravity inversion using conjugate gradient.
    
    Solves: min ||G*m - d||² + λ * ||L*m||²
    
    where G is the sensitivity matrix, L is a smoothness operator,
    and λ is the regularization parameter.
    
    Parameters
    ----------
    data_dict : dict  Contains mesh_info, rx_locs, d_noisy, std
    
    Returns
    -------
    model_rec : np.ndarray  Recovered density model
    """
    mesh_info = data_dict['mesh_info']
    rx_locs = data_dict['rx_locs']
    d_noisy = data_dict['d_noisy']
    std = data_dict['std']
    
    print("[RECON] Building sensitivity matrix ...")
    G = build_sensitivity_matrix(mesh_info, rx_locs)
    
    n_rx, n_cells = G.shape
    nx, ny, nz = mesh_info['shape_cells']
    
    # Weight data by inverse of standard deviation
    Wd = np.diag(1.0 / std)
    Gw = Wd @ G
    dw = Wd @ d_noisy
    
    # Build smoothness regularization matrix (first-order differences)
    # Using simple identity + gradient penalty
    print("[RECON] Building regularization matrix ...")
    
    # Depth weighting: cells deeper get less penalty (to counteract depth decay)
    cc = mesh_info['cell_centers']
    z_weights = np.abs(cc[:, 2]) ** 0.5  # depth weighting
    z_weights = z_weights / z_weights.max()
    z_weights = np.clip(z_weights, 0.1, 1.0)
    
    # Regularization parameter (chosen empirically)
    alpha = 1e-2
    
    # Build Laplacian-like smoothness operator
    L = np.zeros((n_cells, n_cells))
    
    for i in range(n_cells):
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)
        
        count = 0
        # X neighbors
        if ix > 0:
            L[i, i - 1] = -1
            count += 1
        if ix < nx - 1:
            L[i, i + 1] = -1
            count += 1
        # Y neighbors
        if iy > 0:
            L[i, i - nx] = -1
            count += 1
        if iy < ny - 1:
            L[i, i + nx] = -1
            count += 1
        # Z neighbors
        if iz > 0:
            L[i, i - nx * ny] = -1
            count += 1
        if iz < nz - 1:
            L[i, i + nx * ny] = -1
            count += 1
        
        L[i, i] = max(count, 1)
    
    # Apply depth weighting to regularization
    Wz = np.diag(z_weights)
    
    # Normal equations: (G'WdG + alpha*L'WzL) m = G'Wd*d
    print("[RECON] Solving normal equations ...")
    
    GtG = Gw.T @ Gw
    LtL = L.T @ Wz @ L
    
    A = GtG + alpha * LtL
    b = Gw.T @ dw
    
    # Solve using conjugate gradient
    from scipy.sparse.linalg import cg
    from scipy.linalg import cho_factor, cho_solve
    
    try:
        # Try Cholesky factorization (faster for small problems)
        c, low = cho_factor(A)
        model_rec = cho_solve((c, low), b)
    except Exception:
        # Fall back to CG
        model_rec, info = cg(A, b, maxiter=100, tol=1e-6)
        if info != 0:
            print(f"[RECON] CG did not converge, info={info}")
    
    print(f"[RECON] Recovered model range: [{model_rec.min():.4f}, {model_rec.max():.4f}] g/cm³")
    
    return model_rec
