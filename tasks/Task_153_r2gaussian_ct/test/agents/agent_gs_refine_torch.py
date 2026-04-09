import matplotlib

matplotlib.use('Agg')

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def gs_refine_torch(target_np, N=800, iters=1200, lr=0.008):
    """Fit N 2D anisotropic Gaussians to a target image via PyTorch autograd."""
    H, W = target_np.shape
    device = torch.device('cpu')
    target = torch.tensor(target_np, dtype=torch.float32, device=device)

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij')

    flat = np.clip(target_np.flatten(), 0, None)
    prob = flat / (flat.sum() + 1e-12)
    rng = np.random.RandomState(42)
    ng, nr = int(N * 0.8), N - int(N * 0.8)
    idx = rng.choice(len(prob), size=ng, p=prob, replace=True)
    iy, ix = np.unravel_index(idx, (H, W))
    py = np.concatenate([iy / H * 2 - 1 + rng.normal(0, 0.01, ng),
                         rng.uniform(-0.9, 0.9, nr)])
    px = np.concatenate([ix / W * 2 - 1 + rng.normal(0, 0.01, ng),
                         rng.uniform(-0.9, 0.9, nr)])
    init_pos = np.stack([px, py], axis=1).astype(np.float32)

    pos = torch.tensor(init_pos, device=device, requires_grad=True)
    amp_raw = torch.zeros(N, device=device, requires_grad=True)
    sig_raw = torch.full((N, 2), -2.0, device=device, requires_grad=True)
    rot = torch.zeros(N, device=device, requires_grad=True)

    opt = torch.optim.Adam([
        {'params': [pos], 'lr': lr * 2},
        {'params': [amp_raw], 'lr': lr},
        {'params': [sig_raw], 'lr': lr * 0.5},
        {'params': [rot], 'lr': lr * 0.3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.01)

    best_loss, best_img = 1e9, None

    for it in range(iters):
        opt.zero_grad()
        amp = F.softplus(amp_raw)
        sig = F.softplus(sig_raw) * 0.10 + 0.005

        cx = pos[:, 0].view(N, 1, 1)
        cy = pos[:, 1].view(N, 1, 1)
        dx = gx.unsqueeze(0) - cx
        dy = gy.unsqueeze(0) - cy
        cr = torch.cos(rot).view(N, 1, 1)
        sr = torch.sin(rot).view(N, 1, 1)
        xr = cr * dx + sr * dy
        yr = -sr * dx + cr * dy
        sx = sig[:, 0].view(N, 1, 1)
        sy = sig[:, 1].view(N, 1, 1)
        a = amp.view(N, 1, 1)

        rendered = (a * torch.exp(-0.5 * (xr**2 / (sx**2 + 1e-6)
                                          + yr**2 / (sy**2 + 1e-6)))).sum(dim=0)
        rendered = torch.clamp(rendered, 0, None)

        loss_mse = F.mse_loss(rendered, target)
        loss_l1 = F.l1_loss(rendered, target)
        tv = (torch.mean(torch.abs(rendered[1:] - rendered[:-1]))
              + torch.mean(torch.abs(rendered[:, 1:] - rendered[:, :-1])))
        loss = 0.8 * loss_mse + 0.2 * loss_l1 + 2e-4 * tv

        loss.backward()
        opt.step()
        sched.step()

        lv = loss_mse.item()
        if lv < best_loss:
            best_loss = lv
            best_img = rendered.detach().cpu().numpy().copy()

        if (it + 1) % 100 == 0:
            print(f"      GS iter {it+1}/{iters}: mse={lv:.6f}")

    return best_img
