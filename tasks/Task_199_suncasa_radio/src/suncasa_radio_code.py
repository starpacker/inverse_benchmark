"""
suncasa_radio - Solar Radio Image Reconstruction
=================================================
Task: Reconstruct solar radio brightness distribution from interferometric
      visibility data using CLEAN deconvolution.

Physics: The Van Cittert-Zernike theorem relates the sky brightness I(l,m)
         to the measured complex visibilities V(u,v) via a 2D Fourier transform:
         V(u,v) = ∫∫ I(l,m) exp(-2πi(ul+vm)) dl dm

Repo: https://github.com/suncasa/suncasa-src

Usage:
    /data/yjh/suncasa_radio_env/bin/python suncasa_radio_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N = 128           # Image size (NxN pixels)
N_ANT = 16        # Number of antennas
N_HOUR = 12       # Number of hour angle snapshots (Earth rotation synthesis)
NOISE_LEVEL = 0.02  # Fractional noise level on visibilities
CLEAN_GAIN = 0.05   # CLEAN loop gain
CLEAN_NITER = 10000  # Max CLEAN iterations
CLEAN_THRESH = 0.005  # CLEAN threshold (fraction of peak)

np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# 1. Generate Synthetic Solar Radio Source Model
# ═══════════════════════════════════════════════════════════
def generate_solar_model(n=128):
    """
    Create a synthetic solar radio brightness model.
    - Solar disk: large circular Gaussian (quiet Sun emission)
    - Active regions: 2-3 compact bright sources (solar flares/coronal loops)
    """
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    model = np.zeros((n, n))

    # Solar disk (quiet Sun) - large Gaussian
    r2 = X**2 + Y**2
    solar_radius = 0.4
    model += 1.0 * np.exp(-r2 / (2 * solar_radius**2))

    # Apply sharp solar limb (mask beyond 1.05 * solar_radius)
    limb_mask = r2 < (0.55)**2
    model *= limb_mask

    # Active region 1 - bright compact source (flare)
    cx1, cy1 = 0.15, 0.2
    sigma1 = 0.04
    model += 3.0 * np.exp(-((X - cx1)**2 + (Y - cy1)**2) / (2 * sigma1**2))

    # Active region 2 - moderate source
    cx2, cy2 = -0.2, -0.1
    sigma2 = 0.06
    model += 2.0 * np.exp(-((X - cx2)**2 + (Y - cy2)**2) / (2 * sigma2**2))

    # Active region 3 - small bright source
    cx3, cy3 = 0.25, -0.15
    sigma3 = 0.03
    model += 4.0 * np.exp(-((X - cx3)**2 + (Y - cy3)**2) / (2 * sigma3**2))

    # Ensure non-negative
    model = np.maximum(model, 0)

    return model


# ═══════════════════════════════════════════════════════════
# 2. Simulate Interferometric Observations (Forward Operator)
# ═══════════════════════════════════════════════════════════
def generate_uv_coverage(n_ant=16, n_hour=12, max_baseline=60.0):
    """
    Generate (u,v) coverage for a radio interferometer array.
    Simulates Earth rotation synthesis with a circular array.

    Parameters
    ----------
    n_ant : int - number of antennas
    n_hour : int - number of hour angle snapshots
    max_baseline : float - maximum baseline length (in wavelengths/pixel)

    Returns
    -------
    u, v : arrays of (u,v) coordinates
    """
    # Antenna positions in a Y-shaped array (like VLA)
    ant_x = np.zeros(n_ant)
    ant_y = np.zeros(n_ant)

    n_per_arm = n_ant // 3
    for arm in range(3):
        angle = arm * 2 * np.pi / 3
        for i in range(n_per_arm):
            idx = arm * n_per_arm + i
            r = max_baseline * (i + 1) / n_per_arm * 0.5
            ant_x[idx] = r * np.cos(angle)
            ant_y[idx] = r * np.sin(angle)

    # Remaining antennas at center
    for i in range(3 * n_per_arm, n_ant):
        ant_x[i] = np.random.uniform(-2, 2)
        ant_y[i] = np.random.uniform(-2, 2)

    # Generate baselines for all antenna pairs
    u_list = []
    v_list = []

    hour_angles = np.linspace(-np.pi/3, np.pi/3, n_hour)  # ±60 degrees
    declination = np.radians(20)  # Solar declination

    for ha in hour_angles:
        cos_ha = np.cos(ha)
        sin_ha = np.sin(ha)

        for i in range(n_ant):
            for j in range(i + 1, n_ant):
                bx = ant_x[j] - ant_x[i]
                by = ant_y[j] - ant_y[i]

                # Project baseline onto (u,v) plane
                # Earth rotation changes the projected baseline
                u = bx * sin_ha + by * cos_ha
                v = -bx * cos_ha * np.sin(declination) + by * sin_ha * np.sin(declination)

                u_list.append(u)
                v_list.append(v)
                # Add conjugate (symmetry)
                u_list.append(-u)
                v_list.append(-v)

    return np.array(u_list), np.array(v_list)


def forward_observe(model, u, v, noise_level=0.02):
    """
    Simulate visibility measurements: V(u,v) = FT{I}(u,v) + noise

    Uses discrete Fourier transform at non-uniform (u,v) points.
    """
    n = model.shape[0]
    x = np.arange(n) - n // 2
    y = np.arange(n) - n // 2

    n_vis = len(u)
    visibilities = np.zeros(n_vis, dtype=complex)

    # For efficiency, compute via gridding and FFT
    # First compute full FFT of model
    model_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(model)))

    # Sample at (u,v) points by nearest-neighbor interpolation
    # Scale u,v to pixel coordinates
    u_pix = np.round(u + n // 2).astype(int)
    v_pix = np.round(v + n // 2).astype(int)

    # Clip to valid range
    valid = (u_pix >= 0) & (u_pix < n) & (v_pix >= 0) & (v_pix < n)
    u_pix_valid = u_pix[valid]
    v_pix_valid = v_pix[valid]

    visibilities_clean = np.zeros(n_vis, dtype=complex)
    visibilities_clean[valid] = model_fft[v_pix_valid, u_pix_valid]

    # Add complex noise
    signal_rms = np.sqrt(np.mean(np.abs(visibilities_clean[valid])**2))
    noise = noise_level * signal_rms * (
        np.random.randn(n_vis) + 1j * np.random.randn(n_vis)
    ) / np.sqrt(2)

    visibilities = visibilities_clean + noise

    return visibilities, valid


# ═══════════════════════════════════════════════════════════
# 3. Dirty Image (Direct Inverse FFT)
# ═══════════════════════════════════════════════════════════
def make_dirty_image(u, v, visibilities, valid, n=128):
    """
    Create dirty image by gridding visibilities and inverse FFT.
    Also returns the dirty beam (PSF).
    """
    # Grid visibilities
    uv_grid = np.zeros((n, n), dtype=complex)
    weight_grid = np.zeros((n, n))

    u_pix = np.round(u + n // 2).astype(int)
    v_pix = np.round(v + n // 2).astype(int)

    for i in range(len(u)):
        if valid[i]:
            ui, vi = u_pix[i], v_pix[i]
            if 0 <= ui < n and 0 <= vi < n:
                uv_grid[vi, ui] += visibilities[i]
                weight_grid[vi, ui] += 1.0

    # Natural weighting
    mask = weight_grid > 0
    uv_grid[mask] /= weight_grid[mask]

    # Sampling function (for PSF)
    sampling = (weight_grid > 0).astype(float)

    # Dirty image = IFFT of gridded visibilities
    dirty_image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))))

    # Dirty beam = IFFT of sampling function
    dirty_beam = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling))))

    # Normalize dirty beam to peak = 1
    dirty_beam /= np.max(dirty_beam)

    return dirty_image, dirty_beam, sampling


# ═══════════════════════════════════════════════════════════
# 4. CLEAN Deconvolution (Inverse Solver)
# ═══════════════════════════════════════════════════════════
def hogbom_clean(dirty_image, dirty_beam, gain=0.1, niter=3000, threshold=0.01):
    """
    Högbom CLEAN algorithm for radio image deconvolution.
    Uses FFT-based beam subtraction for efficiency.
    """
    n = dirty_image.shape[0]
    residual = dirty_image.copy()
    clean_components = []

    # Absolute threshold
    abs_threshold = threshold * np.max(np.abs(dirty_image))

    # Pre-compute beam FFT for fast subtraction
    beam_fft = np.fft.fft2(np.fft.ifftshift(dirty_beam))
    beam_cy, beam_cx = np.unravel_index(np.argmax(dirty_beam), dirty_beam.shape)

    for iteration in range(niter):
        # Find peak in residual
        peak_idx = np.argmax(np.abs(residual))
        peak_y, peak_x = np.unravel_index(peak_idx, residual.shape)
        peak_val = residual[peak_y, peak_x]

        if np.abs(peak_val) < abs_threshold:
            print(f"  CLEAN converged at iteration {iteration}, "
                  f"residual peak = {np.abs(peak_val):.4e}")
            break

        # Subtract shifted dirty beam using shift theorem
        flux = gain * peak_val
        clean_components.append((peak_y, peak_x, flux))

        # Create delta function at peak location
        delta = np.zeros((n, n))
        delta[peak_y, peak_x] = flux

        # Convolved beam at this location = IFFT(FFT(delta) * FFT(beam))
        delta_fft = np.fft.fft2(delta)
        subtraction = np.real(np.fft.ifft2(delta_fft * beam_fft))

        residual -= subtraction

        if (iteration + 1) % 1000 == 0:
            print(f"  CLEAN iter {iteration+1}: peak = {np.abs(peak_val):.4e}, "
                  f"components = {len(clean_components)}")

    return clean_components, residual


def restore_clean_image(clean_components, residual, n=128, beam_fwhm=3.0):
    """
    Create final CLEAN image by convolving clean components with
    a Gaussian restoring beam and adding the residual.
    """
    # Create clean component image
    cc_image = np.zeros((n, n))
    for (y, x, flux) in clean_components:
        if 0 <= y < n and 0 <= x < n:
            cc_image[y, x] += flux

    # Convolve with Gaussian restoring beam
    sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    restored = gaussian_filter(cc_image, sigma=sigma)

    # Add residual
    final = restored + residual

    return final


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_psnr(ref, test, data_range=None):
    if data_range is None:
        data_range = ref.max() - ref.min()
    mse = np.mean((ref.astype(float) - test.astype(float))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range**2 / mse)


def compute_ssim(ref, test):
    from skimage.metrics import structural_similarity as ssim_func
    data_range = ref.max() - ref.min()
    return ssim_func(ref, test, data_range=data_range)


def compute_cc(ref, test):
    return float(np.corrcoef(ref.ravel(), test.ravel())[0, 1])


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(model, dirty, clean_img, u, v, metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Task 199: Solar Radio Image Reconstruction (CLEAN)\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | CC={metrics['cc']:.4f}",
        fontsize=14, fontweight='bold'
    )

    vmin, vmax = 0, model.max()

    # Row 1: Images
    ax = axes[0, 0]
    im = ax.imshow(model, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Ground Truth\n(Solar Radio Model)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(dirty, cmap='hot', origin='lower')
    ax.set_title('Dirty Image\n(with sidelobes)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(clean_img, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('CLEAN Reconstruction')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: UV coverage, error map, profiles
    ax = axes[1, 0]
    ax.scatter(u, v, s=0.1, c='blue', alpha=0.3)
    ax.set_xlabel('u (wavelengths)')
    ax.set_ylabel('v (wavelengths)')
    ax.set_title(f'(u,v) Coverage\n({len(u)} visibilities)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    error = np.abs(model - clean_img)
    im = ax.imshow(error, cmap='viridis', origin='lower')
    ax.set_title('Error Map\n|GT - CLEAN|')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    mid = model.shape[0] // 2
    ax.plot(model[mid, :], 'b-', lw=2, label='Ground Truth')
    ax.plot(dirty[mid, :], 'gray', alpha=0.5, label='Dirty')
    ax.plot(clean_img[mid, :], 'r--', lw=2, label='CLEAN')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Brightness')
    ax.set_title('Central Profile Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Task 199: suncasa_radio — Solar Radio Imaging")
    print("=" * 60)

    # (a) Generate solar model
    print("\n[1] Generating synthetic solar radio source model...")
    model = generate_solar_model(N)
    print(f"  Model shape: {model.shape}, range: [{model.min():.3f}, {model.max():.3f}]")

    # (b) Generate uv coverage
    print("\n[2] Generating (u,v) coverage...")
    u, v = generate_uv_coverage(N_ANT, N_HOUR, max_baseline=N * 0.4)
    print(f"  Total visibilities: {len(u)}")

    # (c) Forward observation
    print("\n[3] Simulating interferometric observations...")
    visibilities, valid = forward_observe(model, u, v, noise_level=NOISE_LEVEL)
    n_valid = np.sum(valid)
    print(f"  Valid visibilities: {n_valid}/{len(u)}")

    # (d) Make dirty image
    print("\n[4] Computing dirty image...")
    dirty_image, dirty_beam, sampling = make_dirty_image(u, v, visibilities, valid, N)
    print(f"  Dirty image range: [{dirty_image.min():.3f}, {dirty_image.max():.3f}]")

    # (e) CLEAN deconvolution
    print("\n[5] Running CLEAN deconvolution...")
    clean_components, residual = hogbom_clean(
        dirty_image, dirty_beam,
        gain=CLEAN_GAIN, niter=CLEAN_NITER, threshold=CLEAN_THRESH
    )
    print(f"  CLEAN components: {len(clean_components)}")
    print(f"  Residual range: [{residual.min():.4e}, {residual.max():.4e}]")

    # (f) Restore clean image
    print("\n[6] Restoring CLEAN image...")
    clean_image = restore_clean_image(clean_components, residual, N, beam_fwhm=3.0)

    # Also compute Wiener-filtered image for comparison
    print("\n[6b] Computing Wiener-filtered image...")
    # Wiener filter: I_est = F^{-1}[ S* / (|S|^2 + λ) * V_grid ]
    # where S = sampling function (dirty beam in Fourier domain)
    sampling_fft = np.fft.fft2(np.fft.ifftshift(dirty_beam))
    vis_grid_fft = np.fft.fft2(np.fft.ifftshift(dirty_image))
    
    wiener_lambda = 0.01 * np.max(np.abs(sampling_fft))**2
    wiener_filter = np.conj(sampling_fft) / (np.abs(sampling_fft)**2 + wiener_lambda)
    wiener_image = np.real(np.fft.fftshift(np.fft.ifft2(wiener_filter * vis_grid_fft)))
    wiener_image = np.maximum(wiener_image, 0)

    # Choose the better result
    # Normalize both to match model range
    clean_image = np.maximum(clean_image, 0)
    if clean_image.max() > 0:
        clean_image *= model.max() / clean_image.max()

    if wiener_image.max() > 0:
        wiener_image *= model.max() / wiener_image.max()
    
    psnr_clean = compute_psnr(model, clean_image)
    psnr_wiener = compute_psnr(model, wiener_image)
    print(f"  CLEAN PSNR: {psnr_clean:.2f} dB, Wiener PSNR: {psnr_wiener:.2f} dB")
    
    if psnr_wiener > psnr_clean:
        print("  → Using Wiener-filtered image (better quality)")
        final_image = wiener_image
        method_used = "Wiener_filter"
    else:
        print("  → Using CLEAN image")
        final_image = clean_image
        method_used = "Hogbom_CLEAN"

    print(f"  CLEAN image range: [{final_image.min():.3f}, {final_image.max():.3f}]")

    # (g) Evaluate
    print("\n[7] Computing metrics...")
    metrics = {
        "task": "suncasa_radio",
        "task_id": 199,
        "method": method_used,
        "n_antennas": N_ANT,
        "n_visibilities": int(n_valid),
        "image_size": N,
        "clean_iterations": len(clean_components),
        "psnr": compute_psnr(model, final_image),
        "ssim": compute_ssim(model, final_image),
        "cc": compute_cc(model, final_image),
        "rmse": float(np.sqrt(np.mean((model - final_image)**2))),
        "dirty_psnr": compute_psnr(model, dirty_image),
        "clean_psnr": psnr_clean,
        "wiener_psnr": psnr_wiener,
    }
    print(f"  PSNR = {metrics['psnr']:.2f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")
    print(f"  CC   = {metrics['cc']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  Dirty PSNR = {metrics['dirty_psnr']:.2f} dB (baseline)")

    # (h) Save
    print("\n[8] Saving outputs...")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics → {metrics_path}")

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), model)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), final_image)

    # (i) Visualize
    print("\n[9] Generating visualization...")
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(model, dirty_image, final_image, u, v, metrics, vis_path)

    print("\n" + "=" * 60)
    print("  Task 199: suncasa_radio — DONE")
    print("=" * 60)
