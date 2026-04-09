import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def aperture_sum(image, row, col, radius):
    """Sum of pixel values inside a circular aperture."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
    return image[mask].sum(), mask

def compute_snr_at_position(image, row, col, fwhm):
    """
    Detection SNR (aperture photometry, following Mawet et al. 2014).

    Signal = sum in planet aperture − mean of reference aperture sums.
    Noise  = std of reference aperture sums around the annulus.
    """
    ny, nx = image.shape
    cy, cx_img = ny // 2, nx // 2
    sep = np.sqrt((row - cy) ** 2 + (col - cx_img) ** 2)
    ap_r = fwhm / 2.0

    signal, _ = aperture_sum(image, row, col, ap_r)

    # Reference apertures at the same radial separation
    n_ref = max(int(2 * np.pi * sep / (2 * ap_r + 1)), 8)
    ref_sums = []
    for k in range(n_ref):
        theta = 2 * np.pi * k / n_ref
        rr = cy + sep * np.sin(theta)
        cc = cx_img + sep * np.cos(theta)
        if np.sqrt((rr - row) ** 2 + (cc - col) ** 2) < 3 * ap_r:
            continue
        if rr < ap_r or rr >= ny - ap_r or cc < ap_r or cc >= nx - ap_r:
            continue
        s, _ = aperture_sum(image, rr, cc, ap_r)
        ref_sums.append(s)

    if len(ref_sums) < 3:
        return np.abs(signal) / (np.abs(signal) * 0.1 + 1e-10)

    noise_std = np.std(ref_sums)
    mean_bg = np.mean(ref_sums)
    snr = (signal - mean_bg) / (noise_std + 1e-10)
    return snr

def compute_snr_map(image):
    """Pixel-wise SNR map using annular noise estimation."""
    ny, nx = image.shape
    cy, cx_img = ny // 2, nx // 2
    yy, xx = np.mgrid[:ny, :nx]
    r_map = np.sqrt((yy - cy) ** 2 + (xx - cx_img) ** 2)

    snr_map = np.zeros_like(image)
    max_r = int(r_map.max()) + 1
    for r in range(3, max_r):
        annulus = (r_map >= r - 1.5) & (r_map < r + 1.5)
        vals = image[annulus]
        if len(vals) > 10:
            std = np.std(vals)
            mean = np.mean(vals)
            if std > 1e-10:
                ring = (r_map >= r - 0.5) & (r_map < r + 0.5)
                snr_map[ring] = (image[ring] - mean) / std
    return snr_map

def find_peak_near(image, row, col, search_radius=10):
    """Find the peak pixel position near (row, col)."""
    ny, nx = image.shape
    r0 = max(0, int(row - search_radius))
    r1 = min(ny, int(row + search_radius + 1))
    c0 = max(0, int(col - search_radius))
    c1 = min(nx, int(col + search_radius + 1))
    sub = image[r0:r1, c0:c1]
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    return r0 + idx[0], c0 + idx[1]

def evaluate_results(
    final_image,
    ground_truth,
    cube,
    angles,
    params,
    save_dir=None,
    vis_path=None
):
    """
    Evaluate reconstruction quality and optionally save results.
    
    Computes metrics:
    - SNR: Detection signal-to-noise ratio at planet position
    - Position error: Distance between detected peak and true position
    - Photometric accuracy: Recovered flux / injected flux
    - PSNR: Peak signal-to-noise ratio vs ground truth
    
    Parameters
    ----------
    final_image : ndarray (image_size, image_size)
        Reconstructed image from inversion
    ground_truth : dict
        Contains 'planet_position', 'planet_flux', 'clean_planet_image'
    cube : ndarray (n_frames, image_size, image_size)
        Original ADI cube (for visualization)
    angles : ndarray (n_frames,)
        Parallactic angles (for visualization)
    params : dict
        Simulation parameters
    save_dir : str, optional
        Directory to save numpy outputs
    vis_path : str, optional
        Path to save visualization figure
    
    Returns
    -------
    metrics : dict containing:
        - 'snr': Detection SNR
        - 'pos_error': Position error in pixels
        - 'photometric_accuracy': Percent accuracy
        - 'psnr': Peak SNR in dB
    """
    gt_row, gt_col = ground_truth["planet_position"]
    clean_planet = ground_truth["clean_planet_image"]
    planet_fwhm = params.get("planet_fwhm", 5.0)
    image_size = params.get("image_size", 101)
    planet_sep = params.get("planet_sep", 30)

    # Detection SNR
    snr = compute_snr_at_position(final_image, gt_row, gt_col, planet_fwhm)

    # Position error
    peak_row, peak_col = find_peak_near(final_image, gt_row, gt_col)
    pos_error = np.sqrt((peak_row - gt_row) ** 2 + (peak_col - gt_col) ** 2)

    # Photometric accuracy
    ap_r = planet_fwhm / 2.0
    recovered, _ = aperture_sum(final_image, gt_row, gt_col, ap_r)
    injected, _ = aperture_sum(clean_planet, gt_row, gt_col, ap_r)
    photo_acc = (recovered / (injected + 1e-10)) * 100.0

    # PSNR
    signal_max = clean_planet.max()
    mse = np.mean((final_image - clean_planet) ** 2)
    psnr = 10.0 * np.log10(signal_max ** 2 / (mse + 1e-10)) if mse > 0 else float("inf")

    metrics = {
        "snr": snr,
        "pos_error": pos_error,
        "photometric_accuracy": photo_acc,
        "psnr": psnr,
    }
    
    # Save outputs if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "gt_output.npy"), clean_planet)
        np.save(os.path.join(save_dir, "recon_output.npy"), final_image)
        np.save(os.path.join(save_dir, "adi_cube.npy"), cube)
        np.save(os.path.join(save_dir, "angles.npy"), angles)
    
    # Create visualization if path provided
    if vis_path is not None:
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        cx = cy = image_size // 2

        # Panel 1: raw frame (log-stretch)
        ax = axes[0, 0]
        frame = cube[len(cube) // 2]
        log_frame = np.log10(np.clip(frame, 1, None))
        im0 = ax.imshow(log_frame, origin="lower", cmap="inferno")
        ax.set_title("Raw ADI Frame (log₁₀ stretch)", fontsize=13, fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.colorbar(im0, ax=ax, label="log₁₀(Counts)", shrink=0.85)
        ax.plot(cx, cy, "w+", ms=12, mew=2, label="Star")
        mid = len(cube) // 2
        a_rad = np.radians(angles[mid])
        ax.plot(cx + planet_sep * np.cos(a_rad), cy + planet_sep * np.sin(a_rad),
                "co", ms=8, mfc="none", mew=1.5, label="Planet (this frame)")
        ax.legend(loc="upper left", fontsize=8)

        # Panel 2: PCA-ADI residual
        ax = axes[0, 1]
        vabs = np.percentile(np.abs(final_image), 99.5)
        if vabs < 1e-10:
            vabs = 1.0
        im1 = ax.imshow(final_image, origin="lower", cmap="RdBu_r",
                        vmin=-vabs, vmax=vabs)
        ax.set_title("PCA-ADI Residual (Mean Combined)", fontsize=13,
                     fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.colorbar(im1, ax=ax, label="Residual flux", shrink=0.85)
        circ = plt.Circle((gt_col, gt_row), 5, fill=False, ec="lime",
                           lw=2, ls="--", label="True planet")
        ax.add_patch(circ)
        ax.legend(loc="upper left", fontsize=9)

        # Panel 3: ground truth
        ax = axes[1, 0]
        im2 = ax.imshow(clean_planet, origin="lower", cmap="hot")
        ax.set_title("Ground Truth Planet Map", fontsize=13, fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        ax.plot(gt_col, gt_row, "c+", ms=15, mew=2,
                label=f"Planet ({gt_row}, {gt_col})")
        ax.legend(loc="upper left", fontsize=9)
        plt.colorbar(im2, ax=ax, label="Flux", shrink=0.85)

        # Panel 4: SNR map
        ax = axes[1, 1]
        snr_map = compute_snr_map(final_image)
        vmax_snr = max(np.percentile(np.abs(snr_map), 99.5), 1)
        im3 = ax.imshow(snr_map, origin="lower", cmap="viridis",
                        vmin=-3, vmax=vmax_snr)
        ax.set_title("SNR Map", fontsize=13, fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        circ2 = plt.Circle((gt_col, gt_row), 5, fill=False, ec="red",
                            lw=2, ls="--",
                            label=f"Planet SNR={metrics['snr']:.1f}")
        ax.add_patch(circ2)
        ax.legend(loc="upper left", fontsize=9)
        plt.colorbar(im3, ax=ax, label="SNR", shrink=0.85)

        fig.suptitle(
            f"Task 161 — PCA-ADI High-Contrast Imaging\n"
            f"SNR={metrics['snr']:.1f}  |  Pos Error={metrics['pos_error']:.1f} px  |  "
            f"Photo Acc={metrics['photometric_accuracy']:.1f}%  |  "
            f"PSNR={metrics['psnr']:.2f} dB",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIS] Saved → {vis_path}")
    
    return metrics
