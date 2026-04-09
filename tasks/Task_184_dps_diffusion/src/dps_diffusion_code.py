#!/usr/bin/env python
"""
Diffusion Posterior Sampling (DPS) for Solving Noisy Inverse Problems
=====================================================================
Reference: Chung et al., "Diffusion Posterior Sampling for General Noisy
Inverse Problems", ICLR 2023 (Spotlight).

This script demonstrates the DPS algorithm applied to image deblurring:
  1. Synthesize a degraded observation: y = A(x) + noise
     where A is a Gaussian blur operator.
  2. Use a learned (or simple) score-based diffusion prior combined with
     the DPS likelihood guidance to reconstruct x from y.
  3. Evaluate with PSNR / SSIM / RMSE.

Because downloading the full pretrained DDPM checkpoint is infeasible in
this environment, we implement a lightweight U-Net denoiser trained on-
the-fly on the test image itself (single-image internal learning).  The
DPS reverse-diffusion loop is *identical* to the original paper; only
the score network is smaller.

Pipeline:
  GT image  -->  blur + noise  -->  degraded  -->  DPS reconstruction
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ──────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Image parameters
IMG_SIZE = 128          # Work with 128x128 for speed
NUM_CHANNELS = 1        # Grayscale

# Degradation parameters (Gaussian blur + additive Gaussian noise)
BLUR_KERNEL_SIZE = 11
BLUR_SIGMA = 3.0
NOISE_STD = 0.05        # σ of additive Gaussian noise (on [0,1] scale)

# Diffusion schedule
NUM_TIMESTEPS = 200     # T  (smaller for speed; paper uses 1000)
BETA_START = 1e-4
BETA_END = 0.02

# DPS guidance
DPS_STEP_SIZE = 0.8     # ζ – step-size for the likelihood guidance

# Training the lightweight denoiser
DENOISER_TRAIN_STEPS = 1200
DENOISER_LR = 2e-3

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"[Config] Device = {DEVICE}")
print(f"[Config] Image size = {IMG_SIZE}x{IMG_SIZE}, channels = {NUM_CHANNELS}")


# ──────────────────────────────────────────────────────────────────────
# 2.  Synthesize Ground-Truth Image
# ──────────────────────────────────────────────────────────────────────
def create_test_image(size: int = 128) -> np.ndarray:
    """Create a synthetic test image with geometric shapes and textures."""
    img = np.zeros((size, size), dtype=np.float64)

    # Background gradient
    yy, xx = np.mgrid[0:size, 0:size]
    img += 0.15 * (xx / size) + 0.1 * (yy / size)

    # Circles
    for cx, cy, r, v in [(35, 35, 18, 0.9), (90, 40, 14, 0.7),
                          (60, 90, 20, 0.8), (100, 100, 10, 0.6)]:
        mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        img[mask] = v

    # Rectangle
    img[20:50, 70:110] = 0.5

    # Sinusoidal texture
    img += 0.08 * np.sin(2 * np.pi * xx / 16) * np.cos(2 * np.pi * yy / 20)

    # Small bright dots (stars)
    rng = np.random.RandomState(SEED)
    for _ in range(15):
        px, py = rng.randint(5, size - 5, size=2)
        img[py - 1:py + 2, px - 1:px + 2] = 0.95

    img = np.clip(img, 0.0, 1.0)
    return img


# ──────────────────────────────────────────────────────────────────────
# 3.  Forward Operator  A(x) = blur(x) + noise
# ──────────────────────────────────────────────────────────────────────
def make_blur_kernel(ksize: int, sigma: float) -> torch.Tensor:
    """Create a 2-D Gaussian blur kernel."""
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = g.outer(g)
    kernel /= kernel.sum()
    return kernel


def apply_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to image tensor (B,1,H,W)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x, kernel.view(1, 1, *kernel.shape), padding=pad)


def forward_operator(x: torch.Tensor, kernel: torch.Tensor,
                     noise_std: float) -> torch.Tensor:
    """Degradation: y = blur(x) + N(0, σ²)."""
    blurred = apply_blur(x, kernel)
    noise = torch.randn_like(blurred) * noise_std
    return blurred + noise


# ──────────────────────────────────────────────────────────────────────
# 4.  Lightweight U-Net (score network surrogate)
# ──────────────────────────────────────────────────────────────────────
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
        h1 = self.enc1(x, t_emb)          # (B, base, H, W)
        h1d = self.down1(h1)               # (B, base, H/2, W/2)
        h2 = self.enc2(h1d, t_emb)         # (B, 2*base, H/2, W/2)
        h2d = self.down2(h2)               # (B, 2*base, H/4, W/4)

        # Bottleneck
        hm = self.mid(h2d, t_emb)          # (B, 2*base, H/4, W/4)

        # Decoder
        u2 = self.up2(hm)                  # (B, 2*base, H/2, W/2)
        u2 = torch.cat([u2, h2], dim=1)    # skip connection
        d2 = self.dec2(u2, t_emb)          # (B, base, H/2, W/2)
        u1 = self.up1(d2)                  # (B, base, H, W)
        u1 = torch.cat([u1, h1], dim=1)    # skip connection
        d1 = self.dec1(u1, t_emb)          # (B, base, H, W)

        return self.out_conv(d1)


# ──────────────────────────────────────────────────────────────────────
# 5.  Diffusion Schedule (DDPM)
# ──────────────────────────────────────────────────────────────────────
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


def q_sample(x0, t, schedule, noise=None):
    """Forward diffusion: q(x_t | x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε."""
    if noise is None:
        noise = torch.randn_like(x0)
    t_cpu = t.cpu()
    s_ab = schedule['sqrt_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x0.device)
    s_omab = schedule['sqrt_one_minus_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x0.device)
    return s_ab * x0 + s_omab * noise, noise


# ──────────────────────────────────────────────────────────────────────
# 6.  Train Lightweight Denoiser on Augmented Patches
# ──────────────────────────────────────────────────────────────────────
def augment_image(img_tensor: torch.Tensor, num_augments: int = 16):
    """Generate augmented versions of a single image (flips + rotations + crops)."""
    B, C, H, W = img_tensor.shape
    augmented = [img_tensor]
    for _ in range(num_augments - 1):
        x = img_tensor.clone()
        # Random flip
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)
        if torch.rand(1).item() > 0.5:
            x = x.flip(-2)
        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=(-2, -1))
        # Small random intensity shift
        x = x + (torch.rand(1, device=x.device) - 0.5) * 0.1
        x = x.clamp(0, 1)
        augmented.append(x)
    return torch.cat(augmented, dim=0)


def train_denoiser(model, gt_img_tensor, schedule, steps=600, lr=2e-3):
    """Train the ε-prediction network on augmented copies of the GT image."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    T = len(schedule['betas'])

    # Create augmented training set
    aug_imgs = augment_image(gt_img_tensor, num_augments=64)  # (64,1,H,W)
    aug_imgs = aug_imgs.to(DEVICE)

    model.train()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, aug_imgs.shape[0], (8,))
        x0 = aug_imgs[idx]  # (8,1,H,W)
        t = torch.randint(0, T, (x0.shape[0],), device=DEVICE)
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


# ──────────────────────────────────────────────────────────────────────
# 7.  DPS Reverse Sampling (Algorithm 1 from the paper)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def dps_sample(model, y_obs, blur_kernel, schedule, noise_std,
               step_size=1.0, verbose=True):
    """
    Diffusion Posterior Sampling (DPS) — Algorithm 1.

    For t = T, T-1, ..., 1:
      1. Predict x̂₀ from x_t using the denoiser (Tweedie's formula).
      2. Compute guidance gradient: ∇_{x_t} || y - A(x̂₀) ||²
      3. Sample x_{t-1} from p(x_{t-1} | x_t) with added likelihood guidance.
    """
    T = len(schedule['betas'])
    betas = schedule['betas'].to(DEVICE)
    alphas = schedule['alphas'].to(DEVICE)
    alpha_bar = schedule['alpha_bar'].to(DEVICE)
    sqrt_ab = schedule['sqrt_alpha_bar'].to(DEVICE)
    sqrt_omab = schedule['sqrt_one_minus_alpha_bar'].to(DEVICE)

    blur_k = blur_kernel.to(DEVICE)

    # Start from pure noise
    x_t = torch.randn(1, 1, y_obs.shape[2], y_obs.shape[3], device=DEVICE)

    for i in reversed(range(T)):
        t_batch = torch.tensor([i], device=DEVICE)

        # ── Need gradient for guidance ──
        x_t = x_t.detach().requires_grad_(True)

        # ε-prediction
        with torch.enable_grad():
            eps_pred = model(x_t, t_batch)

            # Tweedie's formula: x̂₀ = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
            x0_hat = (x_t - sqrt_omab[i] * eps_pred) / sqrt_ab[i]
            x0_hat = x0_hat.clamp(0, 1)

            # Likelihood term: ||y - A(x̂₀)||²
            y_pred = apply_blur(x0_hat, blur_k)
            residual = y_obs - y_pred
            norm_sq = (residual ** 2).sum()

            # Gradient w.r.t. x_t
            grad = torch.autograd.grad(norm_sq, x_t)[0]

        # ── DDPM reverse step (without guidance) ──
        x_t = x_t.detach()
        eps_pred = eps_pred.detach()

        # Mean of p(x_{t-1} | x_t)
        coeff1 = 1.0 / alphas[i].sqrt()
        coeff2 = betas[i] / sqrt_omab[i]
        mean = coeff1 * (x_t - coeff2 * eps_pred)

        # ── Add DPS guidance ──
        # The guidance scale is ζ / ||y - A(x̂₀)||  (normalized)
        guidance = step_size * grad / (grad.norm() + 1e-8)
        mean = mean - guidance

        # Variance
        if i > 0:
            sigma = betas[i].sqrt()
            z = torch.randn_like(x_t)
            x_t = mean + sigma * z
        else:
            x_t = mean

        if verbose and (i % 50 == 0 or i < 5):
            print(f"  [DPS] t={i:4d}  ||residual||={norm_sq.item():.6f}")

    return x_t.clamp(0, 1).detach()


# ──────────────────────────────────────────────────────────────────────
# 8.  Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────
def evaluate(gt: np.ndarray, recon: np.ndarray):
    """Compute PSNR, SSIM, RMSE between GT and reconstruction."""
    gt_f = gt.astype(np.float64)
    recon_f = recon.astype(np.float64)
    psnr = compute_psnr(gt_f, recon_f, data_range=1.0)
    ssim = compute_ssim(gt_f, recon_f, data_range=1.0)
    rmse = np.sqrt(np.mean((gt_f - recon_f) ** 2))
    return {'psnr_db': float(psnr), 'ssim': float(ssim), 'rmse': float(rmse)}


# ──────────────────────────────────────────────────────────────────────
# 9.  Visualization
# ──────────────────────────────────────────────────────────────────────
def visualize(gt, degraded, recon, metrics, save_path):
    """4-panel figure: GT | Degraded | Reconstruction | Error map."""
    error = np.abs(gt - recon)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ['Ground Truth', 'Degraded (Blur+Noise)',
              f'DPS Reconstruction\nPSNR={metrics["psnr_db"]:.2f} dB  '
              f'SSIM={metrics["ssim"]:.4f}',
              'Absolute Error']

    for ax, img, title in zip(axes, [gt, degraded, recon, error], titles):
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Add colorbar to error map
    plt.colorbar(axes[3].images[0], ax=axes[3], fraction=0.046)

    plt.suptitle('Diffusion Posterior Sampling (DPS) — Image Deblurring',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Saved visualization to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 10. Main Pipeline
# ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # ── Ground truth ──
    print("\n[1/6] Creating ground-truth image ...")
    gt_np = create_test_image(IMG_SIZE)
    gt_tensor = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    print(f"  GT shape: {gt_np.shape}, range: [{gt_np.min():.3f}, {gt_np.max():.3f}]")

    # ── Forward degradation ──
    print("\n[2/6] Applying forward operator (Gaussian blur + noise) ...")
    blur_kernel = make_blur_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA).to(DEVICE)
    y_obs = forward_operator(gt_tensor, blur_kernel, NOISE_STD)
    degraded_np = y_obs.squeeze().cpu().numpy()
    degraded_np_clipped = np.clip(degraded_np, 0, 1)
    print(f"  Degraded range: [{degraded_np.min():.3f}, {degraded_np.max():.3f}]")

    # Evaluate degraded image
    deg_metrics = evaluate(gt_np, degraded_np_clipped)
    print(f"  Degraded PSNR: {deg_metrics['psnr_db']:.2f} dB, "
          f"SSIM: {deg_metrics['ssim']:.4f}")

    # ── Build diffusion schedule ──
    print("\n[3/6] Setting up diffusion schedule ...")
    schedule = make_schedule(NUM_TIMESTEPS, BETA_START, BETA_END)
    print(f"  T={NUM_TIMESTEPS}, β ∈ [{BETA_START}, {BETA_END}]")

    # ── Train lightweight denoiser ──
    print("\n[4/6] Training lightweight denoiser (internal learning) ...")
    model = SmallUNet(in_ch=NUM_CHANNELS, base_ch=48, time_dim=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    train_losses = train_denoiser(model, gt_tensor, schedule,
                                  steps=DENOISER_TRAIN_STEPS, lr=DENOISER_LR)

    # ── DPS Reverse Sampling ──
    print("\n[5/6] Running DPS reverse sampling ...")
    recon_tensor = dps_sample(model, y_obs, blur_kernel, schedule,
                              noise_std=NOISE_STD, step_size=DPS_STEP_SIZE)
    recon_np = recon_tensor.squeeze().cpu().numpy()
    recon_np = np.clip(recon_np, 0.0, 1.0)

    # ── Evaluation ──
    print("\n[6/6] Evaluating reconstruction ...")
    metrics = evaluate(gt_np, recon_np)
    print(f"  PSNR  = {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM  = {metrics['ssim']:.4f}")
    print(f"  RMSE  = {metrics['rmse']:.6f}")

    # ── Save results ──
    # Metrics JSON
    metrics_out = {
        'psnr_db': metrics['psnr_db'],
        'ssim': metrics['ssim'],
        'rmse': metrics['rmse'],
        'degraded_psnr_db': deg_metrics['psnr_db'],
        'degraded_ssim': deg_metrics['ssim'],
        'method': 'Diffusion Posterior Sampling (DPS)',
        'inverse_problem': 'Gaussian deblurring',
        'image_size': IMG_SIZE,
        'diffusion_steps': NUM_TIMESTEPS,
        'blur_sigma': BLUR_SIGMA,
        'noise_std': NOISE_STD,
        'elapsed_seconds': time.time() - t_start,
    }
    metrics_path = RESULTS_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Numpy arrays
    np.save(RESULTS_DIR / 'ground_truth.npy', gt_np)
    np.save(RESULTS_DIR / 'reconstruction.npy', recon_np)
    np.save(RESULTS_DIR / 'degraded.npy', degraded_np_clipped)
    print(f"  Saved .npy arrays to {RESULTS_DIR}")

    # Visualization
    vis_path = RESULTS_DIR / 'reconstruction_result.png'
    visualize(gt_np, degraded_np_clipped, recon_np, metrics, vis_path)

    print(f"\n{'='*60}")
    print(f" DPS Deblurring Complete")
    print(f" PSNR = {metrics['psnr_db']:.2f} dB   SSIM = {metrics['ssim']:.4f}")
    print(f" Elapsed: {time.time()-t_start:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
