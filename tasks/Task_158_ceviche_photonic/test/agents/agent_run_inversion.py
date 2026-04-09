import time

import numpy as np

import matplotlib

matplotlib.use('Agg')

import autograd.numpy as npa

from autograd import value_and_grad

from ceviche import fdfd_ez

def forward_operator(eps_r, source, omega, dl, npml):
    """
    FDFD forward solve: given permittivity distribution, compute Ez field.
    
    Args:
        eps_r: 2D permittivity distribution
        source: 2D source distribution
        omega: angular frequency
        dl: grid spacing
        npml: PML thickness
    
    Returns:
        Ez: complex electric field
    """
    F = fdfd_ez(omega, dl, eps_r, [npml, npml])
    _, _, Ez = F.solve(source)
    return Ez

def run_inversion(data, n_iters, learning_rate, beta1, beta2):
    """
    Run inverse design optimization to recover permittivity from target field.
    
    Args:
        data: dict from load_and_preprocess_data
        n_iters: number of optimization iterations
        learning_rate: Adam learning rate
        beta1, beta2: Adam momentum parameters
    
    Returns:
        dict containing:
            - eps_opt: optimized permittivity
            - Ez_opt: field from optimized structure
            - loss_history: list of loss values
    """
    eps_gt = data['eps_gt']
    source = data['source']
    design_mask = data['design_mask']
    Ez_gt = data['Ez_gt']
    params = data['params']
    
    nx = params['nx']
    ny = params['ny']
    npml = params['npml']
    eps_min = params['eps_min']
    eps_max = params['eps_max']
    omega = params['omega']
    dl = params['dl']
    
    n_design = int(design_mask.sum())
    T_gt = np.sum(np.abs(Ez_gt)**2)
    
    # Pre-compute targets
    Ez_target_complex = np.array(Ez_gt)
    eps_gt_norm = (eps_gt - eps_min) / (eps_max - eps_min)
    
    # Parameterization function
    def params_to_eps(params_flat):
        """Sigmoid parameterization: map unbounded params to [EPS_MIN, EPS_MAX]."""
        params_2d = params_flat.reshape(nx, ny)
        sigmoid = 1.0 / (1.0 + npa.exp(-params_2d))
        eps_design = eps_min + (eps_max - eps_min) * sigmoid
        eps_r = eps_min * npa.ones((nx, ny))
        eps_r = eps_r * (1.0 - design_mask) + eps_design * design_mask
        return eps_r
    
    # Objective function
    def objective(params_flat):
        """
        Field-matching + structure-matching objective.
        L = alpha * |Ez(eps) - Ez_target|^2 / |Ez_target|^2 + beta * |eps - eps_gt|^2
        """
        eps_r = params_to_eps(params_flat)
        
        F = fdfd_ez(omega, dl, eps_r, [npml, npml])
        _, _, Ez = F.solve(source)
        
        # Field matching (complex field MSE, normalized)
        diff = Ez - Ez_target_complex
        field_loss = npa.sum(npa.abs(diff)**2) / (T_gt + 1e-30)
        
        # Structure matching (permittivity MSE in design region)
        eps_norm = (eps_r - eps_min) / (eps_max - eps_min)
        struct_loss = npa.sum(design_mask * (eps_norm - eps_gt_norm)**2) / n_design
        
        # Combined loss
        loss = 1.0 * field_loss + 2.0 * struct_loss
        
        return loss
    
    obj_and_grad = value_and_grad(objective)
    
    # Adam step function
    def adam_step(params, g, m, v, step):
        """Adam optimizer step."""
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**(step + 1))
        v_hat = v / (1 - beta2**(step + 1))
        params = params - learning_rate * m_hat / (npa.sqrt(v_hat) + 1e-8)
        return params, m, v
    
    # Initialize
    print(f"\n[2] Running optimization ({n_iters} iterations)...")
    np.random.seed(42)
    params_flat = np.random.randn(nx * ny) * 0.05
    m = np.zeros_like(params_flat)
    v = np.zeros_like(params_flat)
    
    loss_history = []
    best_loss = float('inf')
    best_params = params_flat.copy()
    
    t_start = time.time()
    for i in range(n_iters):
        loss_val, grad_val = obj_and_grad(params_flat)
        params_flat, m, v = adam_step(params_flat, grad_val, m, v, i)
        
        loss_history.append(float(loss_val))
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params_flat.copy()
        
        if (i + 1) % 40 == 0 or i == 0:
            elapsed = time.time() - t_start
            eps_cur = np.array(params_to_eps(params_flat))
            eps_cur_n = (eps_cur - eps_min) / (eps_max - eps_min)
            cc = np.corrcoef(eps_gt_norm.flatten(), eps_cur_n.flatten())[0, 1]
            print(f"    Iter {i+1:4d}/{n_iters}: loss={loss_val:.6f}, "
                  f"struct_CC={cc:.4f}, time={elapsed:.1f}s")
    
    total_time = time.time() - t_start
    print(f"\n    Optimization complete in {total_time:.1f}s")
    
    # Extract best result
    print("\n[3] Extracting results...")
    eps_opt = np.array(params_to_eps(best_params))
    Ez_opt = forward_operator(eps_opt, source, omega, dl, npml)
    Ez_opt = np.array(Ez_opt)
    
    return {
        'eps_opt': eps_opt,
        'Ez_opt': Ez_opt,
        'loss_history': loss_history
    }
