import numpy as np
import torch
from tqdm import tqdm

# --- 1. Regularizer Factory ---
def make_regularizer(tv_param, value_range_param, sparse_param, ROI, step_size, device):
    s0, e0, s1, e1, s2, e2 = ROI
    min_val = value_range_param[0]
    max_val = value_range_param[1]
    
    def value_range_regu(x):
        # Clamp values to physical range
        x[s0:e0, s1:e1, s2:e2] = torch.clamp(x[s0:e0, s1:e1, s2:e2], min=min_val, max=max_val)
        return x

    def sparse_regu(z):
        # CRITICAL FIX: Handle NoneType for sparse_param
        if sparse_param is None:
            return z
        # CRITICAL FIX: Scale threshold by step_size
        thres = sparse_param * step_size
        return torch.sign(z) * torch.max(torch.abs(z) - thres, torch.zeros_like(z))

    if tv_param[0] is None:
        return lambda x: sparse_regu(value_range_regu(x))
    else:
        tau = tv_param[0]
        step_num = tv_param[1]
        gamma = 1 / (12 * tau)
        
        # Finite Difference Gradient
        def op_grad(x):
            g = torch.zeros(x.shape + (3,), dtype=torch.float32, device=device)
            g[:-1, :, :, 0] = x[1:, :, :] - x[:-1, :, :]
            g[:, :-1, :, 1] = x[:, 1:, :] - x[:, :-1, :]
            g[:, :, :-1, 2] = x[:, :, 1:] - x[:, :, :-1]
            return g
            
        # Divergence (Adjoint of Gradient)
        def op_div(g):
            x = torch.zeros(g.shape[:-1], dtype=torch.float32, device=device)
            tmp = x.clone()
            tmp[1:-1, :, :] = g[1:-1, :, :, 0] - g[:-2, :, :, 0]
            tmp[0, :, :] = g[0, :, :, 0]
            tmp[-1, :, :] = -g[-2, :, :, 0]
            x += tmp
            tmp[:, 1:-1, :] = g[:, 1:-1, :, 1] - g[:, :-2, :, 1]
            tmp[:, 0, :] = g[:, 0, :, 1]
            tmp[:, -1, :] = -g[:, -2, :, 1]
            x += tmp
            tmp[:, :, 1:-1] = g[:, :, 1:-1, 2] - g[:, :, :-2, 2]
            tmp[:, :, 0] = g[:, :, 0, 2]
            tmp[:, :, -1] = -g[:, :, -2, 2]
            x += tmp
            return -x
            
        def proj_grad(g):
            norm = torch.linalg.norm(g, dim=-1)
            norm[norm < 1] = 1
            norm = norm.reshape(g.shape[:-1] + (1,))
            return g / norm

        # Chambolle-Pock / Primal-Dual TV Solver
        def fista_regu(z):
            g_1 = op_grad(z)
            d = g_1.clone()
            q_1 = 1
            x = z.clone()
            
            for i in range(step_num):
                term1 = z - tau * op_div(d)
                term2 = value_range_regu(term1)
                term3 = op_grad(term2)
                term4 = d + gamma * term3
                g_2 = proj_grad(term4)
                x = value_range_regu(z - tau * op_div(g_2))
                q_2 = 0.5 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
                d = g_2 + ((q_1 - 1) / q_2) * (g_2 - g_1)
                q_1 = q_2
                g_1 = g_2
            return sparse_regu(x)
            
        return fista_regu

# --- 2. Gradient Computation ---
def compute_bpm_gradient_batched(init_delta_ri, u_in, u_out, batch_size, cos_factor, dz, domain_size, k0, p_kernel, device):
    """
    Compute gradient of the loss with respect to refractive index using adjoint method.
    """
    ol_factor = k0 * dz / cos_factor.unsqueeze(-1)
    p_kernel_expanded = p_kernel.unsqueeze(0)
    bp_kernel = p_kernel.conj().unsqueeze(0)
    
    grad = torch.zeros_like(init_delta_ri)
    delta_ri = init_delta_ri
    total_loss = 0.0
    num_batches = (u_in.shape[0] + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, u_in.shape[0])
        actual_batch_size = end_idx - start_idx
        
        sub_u_in = u_in[start_idx:end_idx, ...]
        sub_u_out = u_out[start_idx:end_idx, ...]
        sub_ol_factor = ol_factor[start_idx:end_idx, ...]
        
        s_fields = torch.zeros((actual_batch_size, init_delta_ri.shape[0], init_delta_ri.shape[1], init_delta_ri.shape[2]), dtype=torch.cfloat, device=device)
        
        # Forward Propagation
        u = sub_u_in.clone()
        for m in range(init_delta_ri.shape[0]):
            # CRITICAL FIX: Use torch.fft.fft2 and ifft2 (not torch.fft/ifft)
            u = torch.fft.ifft2(torch.fft.fft2(u) * p_kernel_expanded)
            s_fields[:, m, ...] = u.clone()
            u = u * torch.exp(1j * sub_ol_factor * delta_ri[m, ...].unsqueeze(0))
        
        r = u - sub_u_out
        batch_loss = r.abs().mean().item()
        total_loss += batch_loss
        
        # Backward Propagation (Adjoint)
        for m in reversed(range(init_delta_ri.shape[0])):
            r = r * torch.exp(-1j * sub_ol_factor * delta_ri[m, ...].unsqueeze(0))
            batch_grad = -1j * sub_ol_factor * s_fields[:, m, ...].conj() * r
            grad[m, ...] += batch_grad.real.sum(dim=0)
            # CRITICAL FIX: Use torch.fft.fft2 and ifft2
            r = torch.fft.ifft2(torch.fft.fft2(r) * bp_kernel)
    
    grad = grad / u_in.shape[0]
    avg_loss = total_loss / num_batches
    
    return grad, avg_loss

# --- 3. Main Loop ---
def run_inversion(preprocessed_data, reconstruction_config, data_config):
    """
    Run the BPM inversion using gradient descent with FISTA acceleration.
    """
    device = preprocessed_data['device']
    u_inlet = preprocessed_data['u_inlet']
    u_outlet = preprocessed_data['u_outlet']
    p_kernel = preprocessed_data['p_kernel']
    bpm_cosFactor = preprocessed_data['bpm_cosFactor']
    resolution = preprocessed_data['resolution']
    domain_size = preprocessed_data['domain_size']
    region_z = preprocessed_data['region_z']
    k0 = preprocessed_data['k0']
    
    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    batch_size = reconstruction_config['batch_size']
    
    roi_config = data_config['ROI']
    s0 = roi_config[0] if roi_config[0] is not None else 0
    e0 = roi_config[1] if roi_config[1] is not None else region_z
    s1 = roi_config[2] if roi_config[2] is not None else 0
    e1 = roi_config[3] if roi_config[3] is not None else domain_size[1]
    s2 = roi_config[4] if roi_config[4] is not None else 0
    e2 = roi_config[5] if roi_config[5] is not None else domain_size[2]
    ROI = (s0, e0, s1, e1, s2, e2)
    
    # CRITICAL: Extract scalar from list if necessary, or pass None
    sparse_val = reconstruction_config['sparse_param'][0] if reconstruction_config['sparse_param'] is not None else None

    regu_func = make_regularizer(
        reconstruction_config['tv_param'],
        reconstruction_config['value_range_param'],
        sparse_val,
        ROI=ROI,
        step_size=step_size,
        device=device
    )
    
    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    print('RI shape', init_delta_ri.shape)
    
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    
    loss_history = []
    
    print(f"Starting reconstruction for {n_iter} iterations...")
    pbar = tqdm(range(n_iter))
    
    for iteration in pbar:
        grad, loss = compute_bpm_gradient_batched(
            s, u_inlet, u_outlet, batch_size, bpm_cosFactor,
            resolution[0], domain_size, k0, p_kernel, device
        )
        
        loss_history.append(loss)
        pbar.set_postfix({'loss': loss}, refresh=False)
        
        with torch.no_grad():
            z = s - grad * step_size
            x_2 = regu_func(z)
            q_2 = 0.5 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            x_1 = x_2
            q_1 = q_2
    
    delta_ri = s.cpu().numpy()
    
    return delta_ri, loss_history