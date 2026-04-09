import matplotlib

matplotlib.use('Agg')

import torch

def q_sample(x0, t, schedule, noise=None):
    """Forward diffusion: q(x_t | x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε."""
    if noise is None:
        noise = torch.randn_like(x0)
    t_cpu = t.cpu()
    s_ab = schedule['sqrt_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x0.device)
    s_omab = schedule['sqrt_one_minus_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x0.device)
    return s_ab * x0 + s_omab * noise, noise
