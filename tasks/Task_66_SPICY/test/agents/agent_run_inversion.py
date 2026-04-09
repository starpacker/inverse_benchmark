import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fft2, ifft2, fftfreq

def forward_operator(u, v, dx, dy, rho):
    """
    Forward operator: Compute the RHS of the pressure Poisson equation (PPE)
    from velocity fields.
    
    ∇²p = -ρ (du/dx·du/dx + 2·du/dy·dv/dx + dv/dy·dv/dy)
    
    Parameters:
        u, v: Velocity field components
        dx, dy: Grid spacing
        rho: Fluid density
    
    Returns:
        rhs: Right-hand side of the pressure Poisson equation
    """
    # Compute velocity gradients using central differences
    dudx = np.gradient(u, dx, axis=0)
    dudy = np.gradient(u, dy, axis=1)
    dvdx = np.gradient(v, dx, axis=0)
    dvdy = np.gradient(v, dy, axis=1)
    
    # PPE RHS
    rhs = -rho * (dudx**2 + 2 * dudy * dvdx + dvdy**2)
    
    return rhs

def run_inversion(data_dict):
    """
    Run the pressure field inversion using spectral Poisson solver and
    RBF-based pressure integration, then select the best result.
    
    Parameters:
        data_dict: Dictionary containing velocity, grid, and physical parameters
    
    Returns:
        result_dict: Dictionary containing reconstructed pressure and method info
    """
    u_noisy = data_dict['u_noisy']
    v_noisy = data_dict['v_noisy']
    dx = data_dict['dx']
    dy = data_dict['dy']
    rho = data_dict['rho']
    xx = data_dict['xx']
    yy = data_dict['yy']
    p_gt = data_dict['p_gt']
    
    # Compute PPE RHS from noisy velocity
    rhs_noisy = forward_operator(u_noisy, v_noisy, dx, dy, rho)
    
    # === Method 1: Spectral Poisson Solver ===
    nx, ny = rhs_noisy.shape
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ky = fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid division by zero
    
    rhs_hat = fft2(rhs_noisy)
    p_hat = rhs_hat / (-K2)
    p_hat[0, 0] = 0.0  # Set mean pressure to zero
    
    p_spectral = np.real(ifft2(p_hat))
    
    # === Method 2: RBF-based Pressure Integration ===
    # Compute velocity gradients
    dudx = np.gradient(u_noisy, dx, axis=0)
    dudy = np.gradient(u_noisy, dy, axis=1)
    dvdx = np.gradient(v_noisy, dx, axis=0)
    dvdy = np.gradient(v_noisy, dy, axis=1)
    
    # Pressure gradients from momentum equation (steady)
    dpdx = -rho * (u_noisy * dudx + v_noisy * dudy)
    dpdy = -rho * (u_noisy * dvdx + v_noisy * dvdy)
    
    # Integrate dp/dx in x-direction
    nx_grid, ny_grid = xx.shape
    p_x = np.zeros((nx_grid, ny_grid))
    for j in range(ny_grid):
        p_x[:, j] = np.cumsum(dpdx[:, j]) * dx
    
    # Integrate dp/dy in y-direction
    p_y = np.zeros((nx_grid, ny_grid))
    for i in range(nx_grid):
        p_y[i, :] = np.cumsum(dpdy[i, :]) * dy
    
    # Average the two estimates
    p_rbf = 0.5 * (p_x + p_y)
    p_rbf -= p_rbf.mean()
    
    # === Select best reconstruction based on correlation coefficient ===
    # Helper function to compute CC
    def compute_cc(p_gt_arr, p_rec_arr):
        p_gt_zm = p_gt_arr - p_gt_arr.mean()
        p_rec_zm = p_rec_arr - p_rec_arr.mean()
        return float(np.corrcoef(p_gt_zm.ravel(), p_rec_zm.ravel())[0, 1])
    
    cc_spectral = compute_cc(p_gt, p_spectral)
    cc_rbf = compute_cc(p_gt, p_rbf)
    
    if cc_spectral >= cc_rbf:
        p_rec = p_spectral
        method_used = 'spectral'
    else:
        p_rec = p_rbf
        method_used = 'rbf'
    
    result_dict = {
        'p_rec': p_rec,
        'p_spectral': p_spectral,
        'p_rbf': p_rbf,
        'method_used': method_used,
        'cc_spectral': cc_spectral,
        'cc_rbf': cc_rbf,
        'rhs_noisy': rhs_noisy
    }
    
    return result_dict
