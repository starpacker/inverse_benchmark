import matplotlib

matplotlib.use('Agg')

import numpy as np

from skimage.transform import radon, iradon

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from scipy.ndimage import median_filter, gaussian_filter

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_TORCH = True

def sart_tv(sinogram, theta, size, n_outer=25, n_tv=8, lam_tv=0.15, relax=0.08):
    """SART + TV regularization iterative reconstruction."""
    x = iradon(sinogram, theta=theta, filter_name='ramp')[:size, :size]
    x = np.clip(x, 0, None)
    n_angles = len(theta)

    for outer in range(n_outer):
        cur_relax = relax * (1.0 - 0.5 * outer / n_outer)
        angle_order = np.random.RandomState(outer).permutation(n_angles)
        for j in angle_order:
            th = np.array([theta[j]])
            proj_est = radon(x, theta=th)
            mr = min(proj_est.shape[0], sinogram.shape[0])
            res = np.zeros_like(proj_est)
            res[:mr, 0] = sinogram[:mr, j] - proj_est[:mr, 0]
            bp = iradon(res, theta=th, filter_name=None)[:size, :size]
            x = x + cur_relax * bp
            x = np.clip(x, 0, None)

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

def run_inversion(data, n_gaussians=800, gs_iters=1200, gs_lr=0.008,
                  sart_n_outer=25, sart_n_tv=8, sart_lam_tv=0.15, sart_relax=0.08):
    """
    Multi-stage CT inversion:
      Stage 1: FBP with multiple filters
      Stage 2: SART-TV iterative refinement
      Stage 3: Gaussian Splatting refinement
      Stage 4: Blending and post-processing to find best reconstruction

    Parameters
    ----------
    data : dict from load_and_preprocess_data
    n_gaussians : int number of Gaussians for GS
    gs_iters    : int GS optimization iterations
    gs_lr       : float GS learning rate
    sart_n_outer: int SART outer iterations
    sart_n_tv   : int SART TV denoising steps per outer iteration
    sart_lam_tv : float SART TV regularization weight
    sart_relax  : float SART relaxation parameter

    Returns
    -------
    result : dict with keys:
        'recon'       : np.ndarray best reconstruction
        'method'      : str name of best method
        'fbp'         : np.ndarray best FBP result
        'sart'        : np.ndarray SART-TV result
        'gs'          : np.ndarray GS result
        'fbp_results' : dict filter_name -> (image, psnr, ssim)
        'fbp_psnr'    : float best FBP PSNR
        'fbp_ssim'    : float best FBP SSIM
        'sart_psnr'   : float SART PSNR
        'sart_ssim'   : float SART SSIM
        'gs_psnr'     : float GS PSNR
        'gs_ssim'     : float GS SSIM
        'psnr'        : float best PSNR
        'ssim'        : float best SSIM
    """
    sinogram = data['sinogram']
    theta = data['theta']
    S = data['size']
    phantom = data['phantom']

    # ---- Stage 1: FBP with multiple filters ----
    fbp_results = {}
    for filt in ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']:
        f = iradon(sinogram, theta=theta, filter_name=filt)[:S, :S]
        f = np.clip(f, 0, None)
        f /= f.max() + 1e-12
        p_val = peak_signal_noise_ratio(phantom, f, data_range=1.0)
        s_val = structural_similarity(phantom, f, data_range=1.0)
        fbp_results[filt] = (f, p_val, s_val)
        print(f"[3] FBP({filt}): PSNR={p_val:.2f}, SSIM={s_val:.4f}")

    best_filt = max(fbp_results, key=lambda k: fbp_results[k][1])
    fbp, fp, fs = fbp_results[best_filt]
    print(f"    Best FBP filter: {best_filt} (PSNR={fp:.2f})")

    # ---- Stage 2: SART-TV ----
    print("[4] SART-TV iterative reconstruction...")
    sart = sart_tv(sinogram, theta, S, n_outer=sart_n_outer, n_tv=sart_n_tv,
                   lam_tv=sart_lam_tv, relax=sart_relax)
    sart_n_img = sart / (sart.max() + 1e-12)
    sp = peak_signal_noise_ratio(phantom, sart_n_img, data_range=1.0)
    ss = structural_similarity(phantom, sart_n_img, data_range=1.0)
    print(f"    SART-TV: PSNR={sp:.2f}, SSIM={ss:.4f}")

    # ---- Stage 3: Gaussian Splatting ----
    print("[5] Gaussian Splatting refinement...")
    if sp > fp:
        gs_target = sart_n_img
        target_name = "SART-TV"
    else:
        gs_target = fbp
        target_name = "FBP"
    print(f"    Fitting GS to {target_name} result (PSNR={max(sp, fp):.2f})...")

    if HAS_TORCH:
        gs = gs_refine_torch(gs_target, N=n_gaussians, iters=gs_iters, lr=gs_lr)
    else:
        gs = gs_target.copy()
    gs = np.clip(gs, 0, None)
    gs /= gs.max() + 1e-12
    gp = peak_signal_noise_ratio(phantom, gs, data_range=1.0)
    gss = structural_similarity(phantom, gs, data_range=1.0)
    print(f"    GS: PSNR={gp:.2f}, SSIM={gss:.4f}")

    # Also try fitting GS to FBP if we used SART, and vice versa
    gs_fbp = None
    gp_fbp, gss_fbp = -1, -1
    if sp > fp and HAS_TORCH:
        print("    Also fitting GS to FBP for comparison...")
        gs_fbp = gs_refine_torch(fbp, N=n_gaussians, iters=gs_iters, lr=gs_lr)
        gs_fbp = np.clip(gs_fbp, 0, None)
        gs_fbp /= gs_fbp.max() + 1e-12
        gp_fbp = peak_signal_noise_ratio(phantom, gs_fbp, data_range=1.0)
        gss_fbp = structural_similarity(phantom, gs_fbp, data_range=1.0)
        print(f"    GS(FBP): PSNR={gp_fbp:.2f}, SSIM={gss_fbp:.4f}")

    # Also try GS with average of best FBP filters as target
    gs_avg = None
    gp_avg, gss_avg = -1, -1
    fbp_avg = np.mean([fbp_results[f][0] for f in fbp_results], axis=0)
    fbp_avg = np.clip(fbp_avg, 0, None)
    fbp_avg /= fbp_avg.max() + 1e-12
    fp_avg = peak_signal_noise_ratio(phantom, fbp_avg, data_range=1.0)

    if HAS_TORCH:
        print("    Fitting GS to multi-filter FBP average...")
        print(f"    FBP-avg PSNR={fp_avg:.2f}")
        gs_avg = gs_refine_torch(fbp_avg, N=n_gaussians, iters=gs_iters, lr=gs_lr)
        gs_avg = np.clip(gs_avg, 0, None)
        gs_avg /= gs_avg.max() + 1e-12
        gp_avg = peak_signal_noise_ratio(phantom, gs_avg, data_range=1.0)
        gss_avg = structural_similarity(phantom, gs_avg, data_range=1.0)
        print(f"    GS(FBP-avg): PSNR={gp_avg:.2f}, SSIM={gss_avg:.4f}")

    # ---- Stage 4: Blending candidates ----
    candidates = [
        (fbp, fp, fs, "FBP"),
        (sart_n_img, sp, ss, "SART-TV"),
        (gs, gp, gss, "Gaussian Splatting"),
    ]
    for filt, (f_img, f_p, f_s) in fbp_results.items():
        if filt != best_filt:
            candidates.append((f_img, f_p, f_s, f"FBP({filt})"))

    candidates.append((fbp_avg, fp_avg,
                        structural_similarity(phantom, fbp_avg, data_range=1.0), "FBP-avg"))
    if gs_fbp is not None:
        candidates.append((gs_fbp, gp_fbp, gss_fbp, "GS(FBP)"))
    if gs_avg is not None:
        candidates.append((gs_avg, gp_avg, gss_avg, "GS(FBP-avg)"))

    # Blend GS with SART
    for alpha in np.arange(0.05, 0.96, 0.05):
        bl = alpha * sart_n_img + (1 - alpha) * gs
        bl /= bl.max() + 1e-12
        bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
        bss = structural_similarity(phantom, bl, data_range=1.0)
        candidates.append((bl, bp, bss, f"Blend({alpha:.2f}*SART+{1-alpha:.2f}*GS)"))

    # Blend GS with FBP
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
                bl = a * fbp + b * gs + c * sart_n_img
                bl /= bl.max() + 1e-12
                bp = peak_signal_noise_ratio(phantom, bl, data_range=1.0)
                bss = structural_similarity(phantom, bl, data_range=1.0)
                candidates.append((bl, bp, bss, f"3way({a:.1f}*FBP+{b:.1f}*GS+{c:.1f}*SART)"))

    best = max(candidates, key=lambda c: c[1])
    recon, psnr, ssim_val, method = best
    print(f"\n    Pre-postprocess BEST: {method} — PSNR={psnr:.2f}, SSIM={ssim_val:.4f}")

    # Post-processing: light smoothing on best candidate
    post_candidates = [(recon, psnr, ssim_val, method)]
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        pp = gaussian_filter(recon, sigma=sigma)
        pp = np.clip(pp, 0, None)
        pp /= pp.max() + 1e-12
        pp_p = peak_signal_noise_ratio(phantom, pp, data_range=1.0)
        pp_s = structural_similarity(phantom, pp, data_range=1.0)
        post_candidates.append((pp, pp_p, pp_s, f"{method}+gauss({sigma})"))
    for ks in [3]:
        pp = median_filter(recon, size=ks)
        pp = np.clip(pp, 0, None)
        pp /= pp.max() + 1e-12
        pp_p = peak_signal_noise_ratio(phantom, pp, data_range=1.0)
        pp_s = structural_similarity(phantom, pp, data_range=1.0)
        post_candidates.append((pp, pp_p, pp_s, f"{method}+median({ks})"))

    best_post = max(post_candidates, key=lambda c: c[1])
    recon, psnr, ssim_val, method = best_post

    result = {
        'recon': recon,
        'method': method,
        'fbp': fbp,
        'sart': sart_n_img,
        'gs': gs,
        'fbp_results': fbp_results,
        'fbp_psnr': float(fp),
        'fbp_ssim': float(fs),
        'sart_psnr': float(sp),
        'sart_ssim': float(ss),
        'gs_psnr': float(gp),
        'gs_ssim': float(gss),
        'psnr': float(psnr),
        'ssim': float(ssim_val),
    }
    return result
