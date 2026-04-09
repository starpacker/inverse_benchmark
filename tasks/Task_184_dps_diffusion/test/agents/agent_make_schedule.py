import matplotlib

matplotlib.use('Agg')

import torch

def make_schedule(T: int, beta_start: float, beta_end: float):
    """Linear β schedule and derived quantities."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        'betas': betas.float(),
        'alphas': alphas.float(),
        'alpha_bar': alpha_bar.float(),
        'sqrt_alpha_bar': alpha_bar.sqrt().float(),
        'sqrt_one_minus_alpha_bar': (1.0 - alpha_bar).sqrt().float(),
    }
