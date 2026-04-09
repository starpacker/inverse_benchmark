import matplotlib

matplotlib.use('Agg')

import torch

import torch.nn.functional as F

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
