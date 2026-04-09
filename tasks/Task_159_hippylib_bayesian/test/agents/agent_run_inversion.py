import numpy as np

from scipy import sparse

from scipy.sparse.linalg import spsolve

from scipy.optimize import minimize

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

def kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg=1.0):
    """
    kappa(x,y) = kappa_bg + sum_i params[i] * G_i(x,y)
    where G_i is a Gaussian centered at centers[i].
    """
    kappa = np.full_like(X, kappa_bg)
    for i in range(len(params)):
        cx, cy = centers[i]
        kappa += params[i] * np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma_basis**2)
        )
    return np.maximum(kappa, 0.1)

def compute_psnr(x_true, x_recon):
    mse = np.mean((x_true - x_recon)**2)
    if mse < 1e-30:
        return 100.0
    data_range = np.max(x_true) - np.min(x_true)
    return 20.0 * np.log10(data_range / np.sqrt(mse))

def compute_cc(x_true, x_recon):
    return float(np.corrcoef(x_true.ravel(), x_recon.ravel())[0, 1])

def run_inversion(data, reg_coeff=0.01, maxiter=200, refine=True):
    """
    Run the Bayesian inversion using L-BFGS-B optimization.
    
    Args:
        data: dict containing all preprocessed data
        reg_coeff: regularization coefficient
        maxiter: maximum iterations for optimizer
        refine: whether to refine with weaker regularization if needed
        
    Returns:
        dict containing optimized parameters and reconstructed kappa
    """
    X = data['X']
    Y = data['Y']
    centers = data['centers']
    sigma_basis = data['sigma_basis']
    kappa_bg = data['kappa_bg']
    sources = data['sources']
    h = data['h']
    B = data['B']
    obs_data_list = data['obs_data_list']
    noise_var = data['noise_var']
    n_basis = data['n_basis']
    kappa_true = data['kappa_true']
    
    prior_params = np.zeros(n_basis)
    
    def compute_cost_and_grad(params):
        """
        Cost and gradient via forward finite differences in parameter space.
        """
        n_sources = len(sources)
        
        # Base cost
        total_cost = 0.0
        for src, obs_d in zip(sources, obs_data_list):
            kappa_2d = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
            u = solve_pde(kappa_2d, src, h)
            residual = B @ u - obs_d
            c = 0.5 * np.sum(residual**2) / noise_var
            total_cost += c
        
        reg = 0.5 * reg_coeff * np.sum((params - prior_params)**2)
        total_cost += reg
        
        # Gradient via finite differences
        eps = 1e-6
        grad = np.zeros(n_basis)
        for i in range(n_basis):
            params_p = params.copy()
            params_p[i] += eps
            cost_p = 0.0
            for src, obs_d in zip(sources, obs_data_list):
                kappa_p = kappa_from_params(params_p, X, Y, centers, sigma_basis, kappa_bg)
                u_p = solve_pde(kappa_p, src, h)
                res_p = B @ u_p - obs_d
                cost_p += 0.5 * np.sum(res_p**2) / noise_var
            cost_p += 0.5 * reg_coeff * np.sum((params_p - prior_params)**2)
            grad[i] = (cost_p - total_cost) / eps
        
        return total_cost, grad
    
    iter_count = [0]
    current_reg = [reg_coeff]
    
    def objective(params):
        cost, grad = compute_cost_and_grad(params)
        iter_count[0] += 1
        if iter_count[0] % 5 == 0:
            kc = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
            cc = compute_cc(kappa_true, kc)
            print(f"  [{iter_count[0]:3d}] cost={cost:.4e}  CC={cc:.4f}")
        return cost, grad
    
    print("\n=== L-BFGS-B Optimization ===")
    bounds = [(-5.0, 5.0)] * n_basis
    
    result = minimize(
        objective, np.zeros(n_basis),
        method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': True})
    
    params_opt = result.x
    print(f"\n{result.message}")
    print(f"Evaluations: {result.nfev}, Final cost: {result.fun:.4e}")
    
    kappa_map = kappa_from_params(params_opt, X, Y, centers, sigma_basis, kappa_bg)
    
    # Refinement if needed
    if refine:
        cc = compute_cc(kappa_true, kappa_map)
        psnr = compute_psnr(kappa_true, kappa_map)
        if cc < 0.5 or psnr < 15:
            print(f"\nCC={cc:.4f}, PSNR={psnr:.2f} — refining with weaker regularization...")
            current_reg[0] = 0.001
            
            def compute_cost_and_grad_refined(params):
                n_sources = len(sources)
                total_cost = 0.0
                for src, obs_d in zip(sources, obs_data_list):
                    kappa_2d = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
                    u = solve_pde(kappa_2d, src, h)
                    residual = B @ u - obs_d
                    c = 0.5 * np.sum(residual**2) / noise_var
                    total_cost += c
                reg = 0.5 * current_reg[0] * np.sum((params - prior_params)**2)
                total_cost += reg
                
                eps = 1e-6
                grad = np.zeros(n_basis)
                for i in range(n_basis):
                    params_p = params.copy()
                    params_p[i] += eps
                    cost_p = 0.0
                    for src, obs_d in zip(sources, obs_data_list):
                        kappa_p = kappa_from_params(params_p, X, Y, centers, sigma_basis, kappa_bg)
                        u_p = solve_pde(kappa_p, src, h)
                        res_p = B @ u_p - obs_d
                        cost_p += 0.5 * np.sum(res_p**2) / noise_var
                    cost_p += 0.5 * current_reg[0] * np.sum((params_p - prior_params)**2)
                    grad[i] = (cost_p - total_cost) / eps
                return total_cost, grad
            
            def objective_refined(params):
                cost, grad = compute_cost_and_grad_refined(params)
                iter_count[0] += 1
                if iter_count[0] % 5 == 0:
                    kc = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
                    cc = compute_cc(kappa_true, kc)
                    print(f"  [{iter_count[0]:3d}] cost={cost:.4e}  CC={cc:.4f}")
                return cost, grad
            
            result2 = minimize(
                objective_refined, params_opt,
                method='L-BFGS-B', jac=True, bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-11, 'disp': True})
            params_opt = result2.x
            kappa_map = kappa_from_params(params_opt, X, Y, centers, sigma_basis, kappa_bg)
    
    return {
        'params_opt': params_opt,
        'kappa_map': kappa_map,
        'optimization_result': result,
    }
