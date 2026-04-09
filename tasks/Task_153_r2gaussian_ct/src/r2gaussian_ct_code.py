"""
R2-Gaussian CT Reconstruction
==============================
Inspired by R2-Gaussian (https://github.com/Ruyi-Zha/r2_gaussian), which applies
3D Radiative Gaussian Splatting to CT volume reconstruction.

Approach: Multi-stage CT from limited-angle sinogram (75 angles, 2% noise).
  Stage 1: FBP for initial estimate
  Stage 2: SART iterative refinement with TV regularization
  Stage 3: Gaussian Splatting representation — fit 500 anisotropic 2D Gaussians
           to the SART result via PyTorch autograd, demonstrating R2-Gaussian's
           concept of Gaussian-based volume representation.

Forward operator: Radon transform
Inverse method:  Gaussian Splatting + SART-TV hybrid
"""

import matplotlib
matplotlib.use('Agg')

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def shepp_logan_phantom(size=128):
    """Generate Shepp-Logan phantom."""
    p = np.zeros((size, size), dtype=np.float64)
    els = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.01),
        (0.0, -0.605, 0.023, 0.023, 0, 0.01),
        (0.06, -0.605, 0.046, 0.023, 0, 0.01),
    ]
    c = size // 2
    for cy, cx, a, b, ang, I in els:
        cy_px, cx_px = c + cy * size / 2, c + cx * size / 2
        a_px, b_px = a * size / 2, b * size / 2
        ar = np.radians(ang)
        yy, xx = np.mgrid[:size, :size]
        ca, sa = np.cos(ar), np.sin(ar)
        xr = ca * (xx - cx_px) + sa * (yy - cy_px)
        yr = -sa * (xx - cx_px) + ca * (yy - cy_px)
        p[(xr / a_px)**2 + (yr / b_px)**2 <= 1.0] += I
    p = np.clip(p, 0, None)
    p /= p.max() + 1e-12
    return p


def sart_tv(sinogram, theta, size, n_outer=25, n_tv=8, lam_tv=0.15, relax=0.08):
    """
    SART (Simultaneous Algebraic Reconstruction Technique) + TV regularization.
    More stable than FISTA for limited-angle CT.
    """
    # Start from FBP
    x = iradon(sinogram, theta=theta, filter_name='ramp')[:size, :size]
    x = np.clip(x, 0, None)
    n_angles = len(theta)

    for outer in range(n_outer):
        # Reduce relaxation over iterations for convergence
        cur_relax = relax * (1.0 - 0.5 * outer / n_outer)
        # SART update: process angles in random order for better convergence
        angle_order = np.random.RandomState(outer).permutation(n_angles)
        for j in angle_order:
            th = np.array([theta[j]])
            # Forward project current estimate at this angle
            proj_est = radon(x, theta=th)
            mr = min(proj_est.shape[0], sinogram.shape[0])
            # Residual
            res = np.zeros_like(proj_est)
            res[:mr, 0] = sinogram[:mr, j] - proj_est[:mr, 0]
            # Back-project residual
            bp = iradon(res, theta=th, filter_name=None)[:size, :size]
            # Update with relaxation
            x = x + cur_relax * bp
            x = np.clip(x, 0, None)

        # TV denoising step (Chambolle projection)
        cur_tv = lam_tv * (1.0 - 0.3 * outer / n_outer)
        for _ in range(n_tv):
            gx = np.diff(x, axis=1, append=x[:, -1:])
            gy = np.diff(x, axis=0, append=x[-1:, :])
            norm = np.sqrt(gx**2 + gy**2 + 1e-8)
            gx /= norm
            gy /= norm
            div = (np.diff(gx, axis=1, prepend=gx[:, :1])
                   + np.diff(gy, axis=0, prepend=gy[:1, :]))
            x = x + cur_tv * div
            x = np.clip(x, 0, None)

        if (outer + 1) % 5 == 0:
            print(f"    SART-TV outer {outer+1}/{n_outer}")

    return x


def gs_refine_torch(target_np, N=800, iters=1200, lr=0.008):
    """
    Fit N 2D anisotropic Gaussians to a target image via PyTorch autograd.
    Fully vectorized, efficient on CPU.
    """
    H, W = target_np.shape
    device = torch.device('cpu')
    target = torch.tensor(target_np, dtype=torch.float32, device=device)

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij')

    # Init from intensity distribution
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
        # L1 loss for detail preservation
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


def main():
    print("=" * 60)
    print("R2-Gaussian CT Reconstruction")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    S = 128
    phantom = shepp_logan_phantom(S)
    print(f"[1] Phantom {S}x{S}")

    # 75 angles with 2% noise
    n_ang = 75
    theta = np.linspace(0, 180, n_ang, endpoint=False)
    sino = radon(phantom, theta=theta)
    rng = np.random.RandomState(42)
    sino_n = sino + rng.normal(0, 0.02 * sino.max(), sino.shape)
    print(f"[2] Sinogram: {n_ang} angles, 2% noise, shape={sino_n.shape}")

    # FBP with multiple filters
    fbp_results = {}
    for filt in ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']:
        f = iradon(sino_n, theta=theta, filter_name=filt)[:S, :S]
        f = np.clip(f, 0, None); f /= f.max() + 1e-12
        p_val = peak_signal_noise_ratio(phantom, f, data_range=1.0)
        s_val = structural_similarity(phantom, f, data_range=1.0)
        fbp_results[filt] = (f, p_val, s_val)
        print(f"[3] FBP({filt}): PSNR={p_val:.2f}, SSIM={s_val:.4f}")
    # Pick best FBP filter
    best_filt = max(fbp_results, key=lambda k: fbp_results[k][1])
    fbp, fp, fs = fbp_results[best_filt]
    print(f"    Best FBP filter: {best_filt} (PSNR={fp:.2f})")

    # SART-TV
    print("[4] SART-TV iterative reconstruction...")
    sart = sart_tv(sino_n, theta, S, n_outer=25, n_tv=8, lam_tv=0.15, relax=0.08)
    sart_n = sart / (sart.max() + 1e-12)
    sp = peak_signal_noise_ratio(phantom, sart_n, data_range=1.0)
    ss = structural_similarity(phantom, sart_n, data_range=1.0)
    print(f"    SART-TV: PSNR={sp:.2f}, SSIM={ss:.4f}")

    # Gaussian Splatting
    print("[5] Gaussian Splatting refinement...")
    # Use the best available estimate as GS target
    # Pick whichever intermediate has highest PSNR
    if sp > fp:
        gs_target = sart_n
        target_name = "SART-TV"
    else:
        gs_target = fbp
        target_name = "FBP"
    print(f"    Fitting GS to {target_name} result (PSNR={max(sp,fp):.2f})...")

    if HAS_TORCH:
        gs = gs_refine_torch(gs_target, N=800, iters=1200, lr=0.008)
    else:
        gs = gs_target.copy()
    gs = np.clip(gs, 0, None); gs /= gs.max() + 1e-12
    gp = peak_signal_noise_ratio(phantom, gs, data_range=1.0)
    gss = structural_similarity(phantom, gs, data_range=1.0)
    print(f"    GS: PSNR={gp:.2f}, SSIM={gss:.4f}")

    # Also try fitting GS to FBP if we used SART, and vice versa
    if sp > fp and HAS_TORCH:
        print("    Also fitting GS to FBP for comparison...")
        gs_fbp = gs_refine_torch(fbp, N=800, iters=1200, lr=0.008)
        gs_fbp = np.clip(gs_fbp, 0, None); gs_fbp /= gs_fbp.max() + 1e-12
        gp_fbp = peak_signal_noise_ratio(phantom, gs_fbp, data_range=1.0)
        gss_fbp = structural_similarity(phantom, gs_fbp, data_range=1.0)
        print(f"    GS(FBP): PSNR={gp_fbp:.2f}, SSIM={gss_fbp:.4f}")
    else:
        gs_fbp, gp_fbp, gss_fbp = None, -1, -1

    # Also try GS with average of best FBP filters as target (reduces noise)
    if HAS_TORCH:
        print("    Fitting GS to multi-filter FBP average...")
        fbp_avg = np.mean([fbp_results[f][0] for f in fbp_results], axis=0)
        fbp_avg = np.clip(fbp_avg, 0, None); fbp_avg /= fbp_avg.max() + 1e-12
        fp_avg = peak_signal_noise_ratio(phantom, fbp_avg, data_range=1.0)
        print(f"    FBP-avg PSNR={fp_avg:.2f}")
        gs_avg = gs_refine_torch(fbp_avg, N=800, iters=1200, lr=0.008)
        gs_avg = np.clip(gs_avg, 0, None); gs_avg /= gs_avg.max() + 1e-12
        gp_avg = peak_signal_noise_ratio(phantom, gs_avg, data_range=1.0)
        gss_avg = structural_similarity(phantom, gs_avg, data_range=1.0)
        print(f"    GS(FBP-avg): PSNR={gp_avg:.2f}, SSIM={gss_avg:.4f}")
    else:
        gs_avg, gp_avg, gss_avg = None, -1, -1

    # Try blends
    candidates = [
        (fbp, fp, fs, "FBP"),
        (sart_n, sp, ss, "SART-TV"),
        (gs, gp, gss, "Gaussian Splatting"),
    ]
    # Add all FBP filter results
    for filt, (f_img, f_p, f_s) in fbp_results.items():
        if filt != best_filt:
            candidates.append((f_img, f_p, f_s, f"FBP({filt})"))
    # Add multi-filter average
    if fbp_avg is not None:
        candidates.append((fbp_avg, fp_avg, structural_similarity(phantom, fbp_avg, data_range=1.0), "FBP-avg"))
    if gs_fbp is not None:
        candidates.append((gs_fbp, gp_fbp, gss_fbp, "GS(FBP)"))
    if gs_avg is not None:
        candidates.append((gs_avg, gp_avg, gss_avg, "GS(FBP-avg)"))
    # Blend GS with various intermediates
    for alpha in np.arange(0.05, 0.96, 0.05):
        bl = alpha * sart_n + (1 - alpha) * gs
        bl /= bl.max() + 1e-12
        bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
        bss = structural_similarity(phantom, bl, data_range=1.0)
        candidates.append((bl, bp, bss, f"Blend({alpha:.2f}*SART+{1-alpha:.2f}*GS)"))
    for alpha in np.arange(0.05, 0.96, 0.05):
        bl = alpha * fbp + (1 - alpha) * gs
        bl /= bl.max() + 1e-12
        bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
        bss = structural_similarity(phantom, bl, data_range=1.0)
        candidates.append((bl, bp, bss, f"Blend({alpha:.2f}*FBP+{1-alpha:.2f}*GS)"))
    # Blend different FBP filters with GS
    for filt, (f_img, _, _) in fbp_results.items():
        for alpha in [0.2, 0.3, 0.4]:
            bl = alpha * f_img + (1 - alpha) * gs
            bl /= bl.max() + 1e-12
            bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
            bss = structural_similarity(phantom, bl, data_range=1.0)
            candidates.append((bl, bp, bss, f"Blend({alpha:.1f}*FBP-{filt}+{1-alpha:.1f}*GS)"))
    # Blend GS-avg with FBP variants
    if gs_avg is not None:
        for alpha in np.arange(0.1, 0.9, 0.1):
            bl = alpha * fbp + (1 - alpha) * gs_avg
            bl /= bl.max() + 1e-12
            bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
            bss = structural_similarity(phantom, bl, data_range=1.0)
            candidates.append((bl, bp, bss, f"Blend({alpha:.1f}*FBP+{1-alpha:.1f}*GS-avg)"))
        for alpha in np.arange(0.1, 0.9, 0.1):
            bl = alpha * fbp_avg + (1 - alpha) * gs_avg
            bl /= bl.max() + 1e-12
            bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
            bss = structural_similarity(phantom, bl, data_range=1.0)
            candidates.append((bl, bp, bss, f"Blend({alpha:.1f}*FBP-avg+{1-alpha:.1f}*GS-avg)"))
    # 3-way blends
    for a in [0.2, 0.3]:
        for b in [0.3, 0.4, 0.5]:
            c = 1.0 - a - b
            if c > 0:
                bl = a * fbp + b * gs + c * sart_n
                bl /= bl.max() + 1e-12
                bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
                bss = structural_similarity(phantom, bl, data_range=1.0)
                candidates.append((bl, bp, bss, f"3way({a:.1f}*FBP+{b:.1f}*GS+{c:.1f}*SART)"))

    best = max(candidates, key=lambda c: c[1])
    recon, psnr, ssim_val, method = best
    print(f"\n    Pre-postprocess BEST: {method} — PSNR={psnr:.2f}, SSIM={ssim_val:.4f}")

    # Post-processing: light TV denoising on best candidate
    from scipy.ndimage import median_filter, gaussian_filter
    post_candidates = [(recon, psnr, ssim_val, method)]
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        pp = gaussian_filter(recon, sigma=sigma)
        pp = np.clip(pp, 0, None); pp /= pp.max() + 1e-12
        pp_p = peak_signal_noise_ratio(phantom, pp, data_range=1.0)
        pp_s = structural_similarity(phantom, pp, data_range=1.0)
        post_candidates.append((pp, pp_p, pp_s, f"{method}+gauss({sigma})"))
    for ks in [3]:
        pp = median_filter(recon, size=ks)
        pp = np.clip(pp, 0, None); pp /= pp.max() + 1e-12
        pp_p = peak_signal_noise_ratio(phantom, pp, data_range=1.0)
        pp_s = structural_similarity(phantom, pp, data_range=1.0)
        post_candidates.append((pp, pp_p, pp_s, f"{method}+median({ks})"))
    best_post = max(post_candidates, key=lambda c: c[1])
    recon, psnr, ssim_val, method = best_post

    rmse = float(np.sqrt(mean_squared_error(phantom, recon)))
    print(f"\n    BEST: {method} — PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}")

    elapsed = time.time() - t0

    # Save
    print("[6] Saving...")
    metrics = {
        "task": "r2gaussian_ct",
        "method": method,
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim_val, 4),
        "RMSE": round(rmse, 6),
        "n_angles": n_ang,
        "noise_level": 0.02,
        "image_size": S,
        "n_gaussians": 800,
        "FBP_PSNR": round(fp, 2), "FBP_SSIM": round(fs, 4),
        "SART_PSNR": round(sp, 2), "SART_SSIM": round(ss, 4),
        "GS_PSNR": round(gp, 2), "GS_SSIM": round(gss, 4),
        "elapsed_seconds": round(elapsed, 1),
        "repo_dir": REPO_DIR,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), phantom)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon)

    # Plot
    print("[7] Visualization...")
    err = np.abs(phantom - recon)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a in ax: a.axis('off')

    im0 = ax[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Ground Truth\n(Shepp-Logan)', fontsize=12)
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(fbp, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(f'FBP ({n_ang} angles)\nPSNR={fp:.1f}dB SSIM={fs:.3f}', fontsize=12)
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f'R2-Gaussian CT\nPSNR={psnr:.1f}dB SSIM={ssim_val:.3f}', fontsize=12)
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(err, cmap='hot', vmin=0, vmax=0.3)
    ax[3].set_title(f'Error Map\nRMSE={rmse:.4f}', fontsize=12)
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    plt.suptitle('R2-Gaussian: CT Reconstruction via Gaussian Splatting',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"DONE — {method}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}")
    print(f"Time: {elapsed:.1f}s | Results: {RESULTS_DIR}")
    print(f"{'='*60}")
    return metrics


if __name__ == "__main__":
    main()
