import torch


# --- Extracted Dependencies ---

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
        
        u = sub_u_in.clone()
        for m in range(init_delta_ri.shape[0]):
            u = torch.fft.ifft2(torch.fft.fft2(u) * p_kernel_expanded)
            s_fields[:, m, ...] = u.clone()
            u = u * torch.exp(1j * sub_ol_factor * delta_ri[m, ...].unsqueeze(0))
        
        r = u - sub_u_out
        batch_loss = r.abs().mean().item()
        total_loss += batch_loss
        
        for m in reversed(range(init_delta_ri.shape[0])):
            r = r * torch.exp(-1j * sub_ol_factor * delta_ri[m, ...].unsqueeze(0))
            batch_grad = -1j * sub_ol_factor * s_fields[:, m, ...].conj() * r
            grad[m, ...] += batch_grad.real.sum(dim=0)
            r = torch.fft.ifft2(torch.fft.fft2(r) * bp_kernel)
    
    grad = grad / u_in.shape[0]
    avg_loss = total_loss / num_batches
    
    return grad, avg_loss
