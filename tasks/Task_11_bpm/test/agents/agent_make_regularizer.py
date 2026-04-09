import numpy as np
import torch


# --- Extracted Dependencies ---

def make_regularizer(tv_param, value_range_param, sparse_param, ROI, step_size, device):
    s0, e0, s1, e1, s2, e2 = ROI
    min_val = value_range_param[0]
    max_val = value_range_param[1]
    
    def value_range_regu(x):
        x[s0:e0, s1:e1, s2:e2] = torch.clamp(x[s0:e0, s1:e1, s2:e2], min=min_val, max=max_val)
        return x

    def sparse_regu(z):
        if sparse_param is None:
            return z
        thres = sparse_param * step_size
        return torch.sign(z) * torch.max(torch.abs(z) - thres, torch.zeros_like(z))

    if tv_param[0] is None:
        return lambda x: sparse_regu(value_range_regu(x))
    else:
        tau = tv_param[0]
        step_num = tv_param[1]
        gamma = 1 / (12 * tau)
        
        def op_grad(x):
            g = torch.zeros(x.shape + (3,), dtype=torch.float32, device=device)
            g[:-1, :, :, 0] = x[1:, :, :] - x[:-1, :, :]
            g[:, :-1, :, 1] = x[:, 1:, :] - x[:, :-1, :]
            g[:, :, :-1, 2] = x[:, :, 1:] - x[:, :, :-1]
            return g
            
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
