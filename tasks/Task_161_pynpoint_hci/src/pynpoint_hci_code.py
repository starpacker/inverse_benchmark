#!/usr/bin/env python3
"""
Task 161: pynpoint_hci — High-Contrast Imaging (HCI) for exoplanet detection.

Inverse Problem:
    Extract faint exoplanet signals from star PSF-dominated images using
    PCA-based PSF subtraction (Angular Differential Imaging — ADI).

Forward Model:
    True scene (star + planet) → convolve with PSF → rotate by parallactic
    angles → add quasi-static speckle noise + photon / read noise → ADI cube.

Inverse Solver:
    PCA-ADI: fit PCA on the ADI cube to model the quasi-static stellar
    speckle field, subtract it from each frame, de-rotate by −parallactic
    angle, then **mean**-combine the de-rotated residuals to coherently
    build up planet signal while averaging down noise.
"""

import os
import numpy as np
from scipy.ndimage import rotate as ndrotate
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────── CONFIGURATION ────────────────────────────

SEED = 42
N_FRAMES = 100
IMAGE_SIZE = 101
PLANET_SEP = 30           # pixels from star centre
PLANET_CONTRAST = 1e-2    # planet / star peak flux
TOTAL_ROTATION = 90.0     # degrees of field rotation
STAR_FLUX = 1e5
PSF_FWHM = 5.0            # pixels (Gaussian sigma ≈ FWHM / 2.355)
PLANET_FWHM = 5.0
N_PCA_COMPONENTS = 5      # captures star + halo + speckle structure
SPECKLE_AMP = 0.03        # quasi-static speckle amplitude
READ_NOISE = 8.0          # Gaussian read noise σ


# ──────────────────────────── FORWARD MODEL ────────────────────────────

def gaussian_2d(size, cx, cy, flux, fwhm):
    """2-D Gaussian centred at (cx, cy) with peak = `flux`."""
    sigma = fwhm / 2.355
    y, x = np.mgrid[:size, :size]
    return flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))


def make_speckle_field(size, rng, n_modes=25, amp=SPECKLE_AMP):
    """
    Quasi-static speckle field as a sum of random sinusoidal modes.
    Fixed in the pupil plane → identical in every frame.
    Returns a zero-mean, amplitude-controlled field.
    """
    field = np.zeros((size, size))
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_modes):
        kx = rng.uniform(-0.3, 0.3)
        ky = rng.uniform(-0.3, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.3, 1.0)
        field += a * np.cos(2 * np.pi * (kx * xx + ky * yy) + phase)
    field -= field.mean()
    field *= amp / (field.std() + 1e-10)
    return field


def synthesize_adi_cube(
    n_frames=N_FRAMES,
    image_size=IMAGE_SIZE,
    planet_sep=PLANET_SEP,
    planet_contrast=PLANET_CONTRAST,
    total_rotation=TOTAL_ROTATION,
    star_flux=STAR_FLUX,
    psf_fwhm=PSF_FWHM,
    planet_fwhm=PLANET_FWHM,
    read_noise=READ_NOISE,
    seed=SEED,
):
    """
    Synthesise an ADI data cube.

    Returns
    -------
    cube : (n_frames, image_size, image_size)
    angles : (n_frames,)  — parallactic angles [deg]
    ground_truth : dict
    """
    rng = np.random.default_rng(seed)
    cx = cy = image_size // 2
    angles = np.linspace(-total_rotation / 2, total_rotation / 2, n_frames)

    # Stellar PSF (centred, constant across frames)
    star_psf = gaussian_2d(image_size, cx, cy, star_flux, psf_fwhm)

    # Broad stellar halo (Moffat-like, extends to large radii)
    y, x = np.mgrid[:image_size, :image_size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-3
    halo = star_flux * 0.005 / (1.0 + (r / 8.0) ** 2)  # broad halo

    # Quasi-static speckle: modulation of halo
    speckle = make_speckle_field(image_size, rng)
    speckle_pattern = halo * speckle

    # Planet (reference frame: angle = 0, planet along +x axis)
    planet_flux = star_flux * planet_contrast
    planet_row_ref = cy
    planet_col_ref = cx + planet_sep
    clean_planet = gaussian_2d(image_size, planet_col_ref, planet_row_ref,
                               planet_flux, planet_fwhm)

    # Build data cube
    cube = np.zeros((n_frames, image_size, image_size))
    for i, angle in enumerate(angles):
        frame = star_psf + halo + speckle_pattern  # fixed component

        # Planet rotates with parallactic angle
        angle_rad = np.radians(angle)
        px = cx + planet_sep * np.cos(angle_rad)
        py = cy + planet_sep * np.sin(angle_rad)
        planet_img = gaussian_2d(image_size, px, py, planet_flux, planet_fwhm)
        frame = frame + planet_img

        # Photon noise (Poisson) + read noise (Gaussian)
        frame_pos = np.clip(frame, 0.0, None)
        noisy = rng.poisson(frame_pos).astype(np.float64)
        noisy += rng.normal(0, read_noise, frame.shape)
        cube[i] = noisy

    ground_truth = {
        "planet_position": (planet_row_ref, planet_col_ref),
        "planet_flux": planet_flux,
        "clean_planet_image": clean_planet,
    }
    return cube, angles, ground_truth


# ──────────────────────── INVERSE SOLVER: PCA-ADI ──────────────────────

def pca_adi_reduction(cube, angles, n_components=N_PCA_COMPONENTS):
    """
    PCA-based ADI post-processing.

    1. Reshape cube → (n_frames, n_pixels)
    2. PCA models the quasi-static component (star + halo + speckle)
    3. Subtract PCA model from each frame → residuals contain planet + noise
    4. De-rotate each residual frame by −parallactic angle
    5. **Mean** combine → planet coherently accumulates, noise averages down

    Using mean (not median) because the planet only appears at each pixel
    position in a small fraction of frames; median would remove it.
    """
    n_frames, ny, nx = cube.shape
    reshaped = cube.reshape(n_frames, -1)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(reshaped)
    model = pca.inverse_transform(pca.transform(reshaped))
    residuals = (reshaped - model).reshape(n_frames, ny, nx)

    # De-rotate
    derotated = np.zeros_like(residuals)
    for i in range(n_frames):
        derotated[i] = ndrotate(residuals[i], -angles[i], reshape=False, order=3)

    # Mean combine (planet coherent, noise averages)
    final_image = np.mean(derotated, axis=0)
    return final_image, derotated


# ──────────────────────── EVALUATION METRICS ───────────────────────────

def aperture_sum(image, row, col, radius):
    """Sum of pixel values inside a circular aperture."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
    return image[mask].sum(), mask


def compute_snr_at_position(image, row, col, fwhm=PLANET_FWHM):
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


def evaluate(final_image, ground_truth):
    """Compute evaluation metrics."""
    gt_row, gt_col = ground_truth["planet_position"]
    clean_planet = ground_truth["clean_planet_image"]

    # Detection SNR
    snr = compute_snr_at_position(final_image, gt_row, gt_col)

    # Position error
    peak_row, peak_col = find_peak_near(final_image, gt_row, gt_col)
    pos_error = np.sqrt((peak_row - gt_row) ** 2 + (peak_col - gt_col) ** 2)

    # Photometric accuracy
    ap_r = PLANET_FWHM / 2.0
    recovered, _ = aperture_sum(final_image, gt_row, gt_col, ap_r)
    injected, _ = aperture_sum(clean_planet, gt_row, gt_col, ap_r)
    photo_acc = (recovered / (injected + 1e-10)) * 100.0

    # PSNR
    signal_max = clean_planet.max()
    mse = np.mean((final_image - clean_planet) ** 2)
    psnr = 10.0 * np.log10(signal_max ** 2 / (mse + 1e-10)) if mse > 0 else float("inf")

    return {
        "snr": snr,
        "pos_error": pos_error,
        "photometric_accuracy": photo_acc,
        "psnr": psnr,
    }


# ──────────────────────── VISUALIZATION ────────────────────────────────

def visualize(cube, angles, final_image, ground_truth, metrics, save_path):
    """Create a 4-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    gt_row, gt_col = ground_truth["planet_position"]
    cx = cy = IMAGE_SIZE // 2

    # Panel 1: raw frame (log-stretch)
    ax = axes[0, 0]
    frame = cube[len(cube) // 2]
    log_frame = np.log10(np.clip(frame, 1, None))
    im0 = ax.imshow(log_frame, origin="lower", cmap="inferno")
    ax.set_title("Raw ADI Frame (log₁₀ stretch)", fontsize=13, fontweight="bold")
    ax.set_xlabel("x [pixels]"); ax.set_ylabel("y [pixels]")
    plt.colorbar(im0, ax=ax, label="log₁₀(Counts)", shrink=0.85)
    ax.plot(cx, cy, "w+", ms=12, mew=2, label="Star")
    # Mark planet in this frame (angle ≈ 0)
    mid = len(cube) // 2
    a_rad = np.radians(angles[mid])
    ax.plot(cx + PLANET_SEP * np.cos(a_rad), cy + PLANET_SEP * np.sin(a_rad),
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
    ax.set_xlabel("x [pixels]"); ax.set_ylabel("y [pixels]")
    plt.colorbar(im1, ax=ax, label="Residual flux", shrink=0.85)
    circ = plt.Circle((gt_col, gt_row), 5, fill=False, ec="lime",
                       lw=2, ls="--", label="True planet")
    ax.add_patch(circ)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 3: ground truth
    ax = axes[1, 0]
    clean = ground_truth["clean_planet_image"]
    im2 = ax.imshow(clean, origin="lower", cmap="hot")
    ax.set_title("Ground Truth Planet Map", fontsize=13, fontweight="bold")
    ax.set_xlabel("x [pixels]"); ax.set_ylabel("y [pixels]")
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
    ax.set_xlabel("x [pixels]"); ax.set_ylabel("y [pixels]")
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] Saved → {save_path}")


# ──────────────────────── MAIN PIPELINE ────────────────────────────────

def main():
    sandbox = "/data/yjh/pynpoint_hci_sandbox"
    assets = "/data/yjh/website_assets/Task_161_pynpoint_hci"

    print("=" * 70)
    print("  Task 161: pynpoint_hci — PCA-ADI High-Contrast Imaging")
    print("=" * 70)

    # 1. Synthesise
    print("\n[1/4] Synthesising ADI cube …")
    cube, angles, ground_truth = synthesize_adi_cube()
    print(f"       Cube shape        : {cube.shape}")
    print(f"       Angles            : [{angles[0]:.1f}°, {angles[-1]:.1f}°]")
    print(f"       Planet position   : {ground_truth['planet_position']}")
    print(f"       Planet flux       : {ground_truth['planet_flux']:.1f}")

    # 2. PCA-ADI reduction
    print("\n[2/4] Running PCA-ADI reduction (n_components={}) …".format(N_PCA_COMPONENTS))
    final_image, derotated = pca_adi_reduction(cube, angles, N_PCA_COMPONENTS)
    print(f"       Final image shape : {final_image.shape}")
    print(f"       Residual range    : [{final_image.min():.2f}, {final_image.max():.2f}]")

    # 3. Evaluate
    print("\n[3/4] Evaluating …")
    metrics = evaluate(final_image, ground_truth)
    for k, v in metrics.items():
        print(f"       {k:25s}: {v:.4f}")

    # 4. Save
    print("\n[4/4] Saving outputs …")
    gt_output = ground_truth["clean_planet_image"]
    recon_output = final_image

    np.save(os.path.join(sandbox, "gt_output.npy"), gt_output)
    np.save(os.path.join(sandbox, "recon_output.npy"), recon_output)
    np.save(os.path.join(sandbox, "adi_cube.npy"), cube)
    np.save(os.path.join(sandbox, "angles.npy"), angles)
    np.save(os.path.join(assets, "gt_output.npy"), gt_output)
    np.save(os.path.join(assets, "recon_output.npy"), recon_output)

    vis_path = os.path.join(assets, "vis_result.png")
    visualize(cube, angles, final_image, ground_truth, metrics, vis_path)

    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  PSNR         : {metrics['psnr']:.2f} dB")
    print(f"  SNR          : {metrics['snr']:.2f}")
    print(f"  Pos Error    : {metrics['pos_error']:.2f} px")
    print(f"  Photo Acc    : {metrics['photometric_accuracy']:.2f} %")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = main()
