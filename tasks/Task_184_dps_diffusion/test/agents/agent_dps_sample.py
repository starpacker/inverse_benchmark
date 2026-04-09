import matplotlib

matplotlib.use('Agg')

import torch

import torch.nn.functional as F

def apply_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to image tensor (B,1,H,W)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x, kernel.view(1, 1, *kernel.shape), padding=pad)

def dps_sample(model, y_obs, blur_kernel, schedule, device, noise_std,
               step_size=1.0, verbose=True):
    """
    Diffusion Posterior Sampling (DPS) — Algorithm 1.
    """
    T = len(schedule['betas'])
    betas = schedule['betas'].to(device)
    alphas = schedule['alphas'].to(device)
    alpha_bar = schedule['alpha_bar'].to(device)
    sqrt_ab = schedule['sqrt_alpha_bar'].to(device)
    sqrt_omab = schedule['sqrt_one_minus_alpha_bar'].to(device)

    blur_k = blur_kernel.to(device)

    x_t = torch.randn(1, 1, y_obs.shape[2], y_obs.shape[3], device=device)

    for i in reversed(range(T)):
        t_batch = torch.tensor([i], device=device)

        x_t = x_t.detach().requires_grad_(True)

        with torch.enable_grad():
            eps_pred = model(x_t, t_batch)

            x0_hat = (x_t - sqrt_omab[i] * eps_pred) / sqrt_ab[i]
            x0_hat = x0_hat.clamp(0, 1)

            y_pred = apply_blur(x0_hat, blur_k)
            residual = y_obs - y_pred
            norm_sq = (residual ** 2).sum()

            grad = torch.autograd.grad(norm_sq, x_t)[0]

        x_t = x_t.detach()
        eps_pred = eps_pred.detach()

        coeff1 = 1.0 / alphas[i].sqrt()
        coeff2 = betas[i] / sqrt_omab[i]
        mean = coeff1 * (x_t - coeff2 * eps_pred)

        guidance = step_size * grad / (grad.norm() + 1e-8)
        mean = mean - guidance

        if i > 0:
            sigma = betas[i].sqrt()
            z = torch.randn_like(x_t)
            x_t = mean + sigma * z
        else:
            x_t = mean

        if verbose and (i % 50 == 0 or i < 5):
            print(f"  [DPS] t={i:4d}  ||residual||={norm_sq.item():.6f}")

    return x_t.clamp(0, 1).detach()
