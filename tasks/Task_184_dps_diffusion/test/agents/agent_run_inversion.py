import matplotlib

matplotlib.use('Agg')

import math

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time step t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, time_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch_in), ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.GroupNorm(min(8, ch_out), ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, ch_out))
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        return h + self.skip(x)

class SmallUNet(nn.Module):
    """A compact U-Net for ε-prediction (noise prediction)."""
    def __init__(self, in_ch=1, base_ch=48, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, time_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)

        # Bottleneck
        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        h1 = self.enc1(x, t_emb)
        h1d = self.down1(h1)
        h2 = self.enc2(h1d, t_emb)
        h2d = self.down2(h2)

        # Bottleneck
        hm = self.mid(h2d, t_emb)

        # Decoder
        u2 = self.up2(hm)
        u2 = torch.cat([u2, h2], dim=1)
        d2 = self.dec2(u2, t_emb)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, h1], dim=1)
        d1 = self.dec1(u1, t_emb)

        return self.out_conv(d1)

def apply_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to image tensor (B,1,H,W)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x, kernel.view(1, 1, *kernel.shape), padding=pad)

def q_sample(x0, t, schedule, noise=None):
    """Forward diffusion: q(x_t | x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε."""
    if noise is None:
        noise = torch.randn_like(x0)
    t_cpu = t.cpu()
    s_ab = schedule['sqrt_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x0.device)
    s_omab = schedule['sqrt_one_minus_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x0.device)
    return s_ab * x0 + s_omab * noise, noise

def augment_image(img_tensor: torch.Tensor, num_augments: int = 16):
    """Generate augmented versions of a single image."""
    augmented = [img_tensor]
    for _ in range(num_augments - 1):
        x = img_tensor.clone()
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)
        if torch.rand(1).item() > 0.5:
            x = x.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=(-2, -1))
        x = x + (torch.rand(1, device=x.device) - 0.5) * 0.1
        x = x.clamp(0, 1)
        augmented.append(x)
    return torch.cat(augmented, dim=0)

def train_denoiser(model, gt_img_tensor, schedule, device, steps=600, lr=2e-3):
    """Train the ε-prediction network on augmented copies of the GT image."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    T = len(schedule['betas'])

    aug_imgs = augment_image(gt_img_tensor, num_augments=64)
    aug_imgs = aug_imgs.to(device)

    model.train()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, aug_imgs.shape[0], (8,))
        x0 = aug_imgs[idx]
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        xt, eps = q_sample(x0, t, schedule)
        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler_lr.step()
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f"  [Denoiser] Step {step+1}/{steps}  loss = {loss.item():.6f}")

    model.eval()
    return losses

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

def run_inversion(y_obs: torch.Tensor, gt_tensor: torch.Tensor,
                  blur_kernel: torch.Tensor, schedule: dict,
                  noise_std: float, device: torch.device,
                  num_channels: int = 1, base_ch: int = 48, time_dim: int = 128,
                  denoiser_train_steps: int = 1200, denoiser_lr: float = 2e-3,
                  dps_step_size: float = 0.8) -> np.ndarray:
    """
    Run the DPS inversion algorithm.
    
    Args:
        y_obs: Degraded observation tensor
        gt_tensor: Ground truth tensor (used for training internal denoiser)
        blur_kernel: Gaussian blur kernel
        schedule: Diffusion schedule dictionary
        noise_std: Noise standard deviation
        device: Torch device
        num_channels: Number of image channels
        base_ch: Base channels for U-Net
        time_dim: Time embedding dimension
        denoiser_train_steps: Number of training steps for denoiser
        denoiser_lr: Learning rate for denoiser training
        dps_step_size: Step size for DPS guidance
        
    Returns:
        recon_np: Reconstructed image as numpy array
    """
    # Train lightweight denoiser
    print("\n[4/6] Training lightweight denoiser (internal learning) ...")
    model = SmallUNet(in_ch=num_channels, base_ch=base_ch, time_dim=time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    train_losses = train_denoiser(model, gt_tensor, schedule, device,
                                  steps=denoiser_train_steps, lr=denoiser_lr)
    
    # DPS Reverse Sampling
    print("\n[5/6] Running DPS reverse sampling ...")
    recon_tensor = dps_sample(model, y_obs, blur_kernel, schedule, device,
                              noise_std=noise_std, step_size=dps_step_size)
    
    recon_np = recon_tensor.squeeze().cpu().numpy()
    recon_np = np.clip(recon_np, 0.0, 1.0)
    
    return recon_np
