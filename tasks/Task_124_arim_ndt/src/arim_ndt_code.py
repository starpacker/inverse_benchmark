"""
arim_ndt - NDT Ultrasonic Total Focusing Method (TFM) Imaging
=============================================================
Task: Reconstruct defect locations from Full Matrix Capture (FMC) phased array data
Repo: https://github.com/ndtatbristol/arim

Implements TFM (Total Focusing Method) from scratch for ultrasonic phased-array
non-destructive testing. Simulates Full Matrix Capture data with known defect
positions, then reconstructs the image using delay-and-sum beamforming with
analytic signal (Hilbert transform) envelope detection.

Usage:
    /data/yjh/arim_ndt_env/bin/python arim_ndt_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.signal import hilbert

# ============================================================================
# Configuration
# ============================================================================
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# --- Physical parameters ---
N_ELEMENTS = 64           # number of array elements (increased for better focusing)
PITCH = 0.6e-3            # element spacing (m)
FREQ = 5e6                # center frequency (Hz)
C_SOUND = 5900.0          # longitudinal wave speed in steel (m/s)
BANDWIDTH = 0.5           # fractional bandwidth
FS = 50e6                 # sampling frequency (Hz)
N_SAMPLES = 2048          # samples per A-scan
SNR_DB = 40               # signal-to-noise ratio (dB) - increased for cleaner data
N_CYCLES = 3              # number of cycles in toneburst

# --- Defect positions (x_mm, z_mm) ---
DEFECTS_MM = [
    (0.0, 20.0),
    (-5.0, 30.0),
    (3.0, 15.0),
    (7.0, 25.0),
]

# --- Imaging grid ---
X_MIN, X_MAX = -12e-3, 12e-3
Z_MIN, Z_MAX = 5e-3, 40e-3
NX, NZ = 200, 300  # grid resolution (increased for finer detail)


# ============================================================================
# Signal generation
# ============================================================================
def generate_toneburst(fc, n_cycles, fs):
    """Generate a Hanning-windowed tone burst.

    Parameters
    ----------
    fc : float
        Center frequency (Hz).
    n_cycles : int
        Number of cycles.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    t_burst : ndarray
        Time axis for the burst.
    burst : ndarray
        Toneburst waveform.
    """
    duration = n_cycles / fc
    n_pts = int(np.ceil(duration * fs))
    t_burst = np.arange(n_pts) / fs
    burst = np.sin(2 * np.pi * fc * t_burst) * np.hanning(n_pts)
    return t_burst, burst


# ============================================================================
# Forward model: Full Matrix Capture synthesis
# ============================================================================
def compute_element_positions(n_elements, pitch):
    """Return 1-D array of element x-positions centered on 0."""
    return (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch


def synthesize_fmc(defects_m, element_positions, toneburst, fs, n_samples,
                   c, snr_db):
    """Simulate Full Matrix Capture (FMC) data.

    For each transmitter-receiver pair, accumulate the round-trip response
    from every point scatterer with 1/r geometric spreading.

    Parameters
    ----------
    defects_m : list of (x, z) tuples
        Scatterer positions in metres.
    element_positions : ndarray, shape (N,)
        Element x-positions (m).
    toneburst : ndarray
        Reference pulse waveform.
    fs : float
        Sampling frequency (Hz).
    n_samples : int
        Number of time samples per A-scan.
    c : float
        Sound speed (m/s).
    snr_db : float
        Desired signal-to-noise ratio (dB).

    Returns
    -------
    fmc : ndarray, shape (N, N, n_samples)
        Full matrix capture data.
    t_axis : ndarray, shape (n_samples,)
        Time axis (s).
    """
    n_elem = len(element_positions)
    fmc = np.zeros((n_elem, n_elem, n_samples))
    t_axis = np.arange(n_samples) / fs
    burst_len = len(toneburst)

    for dx, dz in defects_m:
        # Distances from each element to this defect
        dist = np.sqrt((element_positions - dx) ** 2 + dz ** 2)  # (N,)

        for tx in range(n_elem):
            d_tx = dist[tx]
            for rx in range(n_elem):
                d_rx = dist[rx]
                delay = (d_tx + d_rx) / c
                amplitude = 1.0 / (d_tx * d_rx) * 1e-3  # geometric spreading
                sample_start = int(round(delay * fs))
                sample_end = sample_start + burst_len
                if sample_end < n_samples:
                    fmc[tx, rx, sample_start:sample_end] += amplitude * toneburst

    # Add white Gaussian noise
    signal_power = np.mean(fmc ** 2)
    if signal_power > 0:
        noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))
        fmc += np.random.randn(*fmc.shape) * noise_std

    return fmc, t_axis


# ============================================================================
# TFM reconstruction (vectorized)
# ============================================================================
def tfm_reconstruct(fmc, element_positions, x_grid, z_grid, fs, c):
    """Total Focusing Method image reconstruction (vectorized).

    Uses numpy broadcasting to avoid Python-level loops over pixels and
    element pairs as much as possible.  The analytic signal (via Hilbert
    transform) is computed once, then linear interpolation into continuous
    delay values gives the coherent sum.

    Parameters
    ----------
    fmc : ndarray, shape (N, N, n_samples)
        Full matrix capture data.
    element_positions : ndarray, shape (N,)
        Element x-positions (m).
    x_grid : ndarray, shape (NX,)
        Pixel x-coordinates (m).
    z_grid : ndarray, shape (NZ,)
        Pixel z-coordinates (m).
    fs : float
        Sampling frequency (Hz).
    c : float
        Sound speed (m/s).

    Returns
    -------
    image : ndarray, shape (NZ, NX)
        TFM image (envelope amplitude).
    """
    n_elem = fmc.shape[0]
    n_samples = fmc.shape[2]

    # Compute analytic signal for each TX-RX pair
    print("  Computing analytic signal (Hilbert transform)...")
    fmc_analytic = np.zeros_like(fmc, dtype=np.complex128)
    for i in range(n_elem):
        fmc_analytic[i, :, :] = hilbert(fmc[i, :, :], axis=-1)

    # Precompute distances from each element to each pixel
    # element_positions: (N,)  x_grid: (NX,)  z_grid: (NZ,)
    # distances shape: (N, NZ, NX)
    print("  Precomputing distance tables...")
    ex = element_positions[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    gx = x_grid[np.newaxis, np.newaxis, :]             # (1, 1, NX)
    gz = z_grid[np.newaxis, :, np.newaxis]              # (1, NZ, 1)
    distances = np.sqrt((ex - gx) ** 2 + gz ** 2)      # (N, NZ, NX)

    # Convert distances to sample indices (fractional)
    # delay_samples[elem, iz, ix] = distance / c * fs
    delay_samples = distances / c * fs  # (N, NZ, NX)

    image = np.zeros((len(z_grid), len(x_grid)))

    print(f"  Delay-and-sum over {n_elem}x{n_elem} element pairs...")
    t0 = time.time()
    for tx in range(n_elem):
        if tx % 8 == 0:
            elapsed = time.time() - t0
            print(f"    TX {tx}/{n_elem}  ({elapsed:.1f}s elapsed)")
        d_tx = delay_samples[tx]  # (NZ, NX)
        for rx in range(n_elem):
            d_rx = delay_samples[rx]  # (NZ, NX)
            total_delay = d_tx + d_rx  # round-trip delay in samples

            # Integer and fractional parts for linear interpolation
            idx = total_delay.astype(np.int64)
            frac = total_delay - idx

            # Mask valid indices
            valid = (idx >= 0) & (idx < n_samples - 1)

            # Safe index (clamp for out-of-range, we'll zero them later)
            idx_safe = np.clip(idx, 0, n_samples - 2)

            # Linearly interpolate analytic signal
            val = ((1.0 - frac) * fmc_analytic[tx, rx, idx_safe] +
                   frac * fmc_analytic[tx, rx, idx_safe + 1])
            val[~valid] = 0.0

            image += np.abs(val)

    elapsed = time.time() - t0
    print(f"  TFM reconstruction completed in {elapsed:.1f}s")
    return image


# ============================================================================
# Ground truth defect map
# ============================================================================
def create_ground_truth_map(defects_m, x_grid, z_grid, spot_sigma=0.5e-3):
    """Create a 2-D Gaussian-spot ground truth defect map.

    Parameters
    ----------
    defects_m : list of (x, z) tuples
        Defect positions in metres.
    x_grid, z_grid : ndarray
        1-D coordinate arrays.
    spot_sigma : float
        Gaussian spot standard deviation (m).

    Returns
    -------
    gt_map : ndarray, shape (NZ, NX)
        Ground truth image with Gaussian spots at defect locations.
    """
    XX, ZZ = np.meshgrid(x_grid, z_grid)
    gt_map = np.zeros_like(XX)
    for dx, dz in defects_m:
        gt_map += np.exp(-((XX - dx) ** 2 + (ZZ - dz) ** 2) / (2 * spot_sigma ** 2))
    gt_map /= gt_map.max()  # normalize to [0, 1]
    return gt_map


# ============================================================================
# Metrics
# ============================================================================
def compute_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-20:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(gt, recon):
    """Structural Similarity Index (simplified, no windowing)."""
    from skimage.metrics import structural_similarity
    return structural_similarity(gt, recon, data_range=gt.max() - gt.min())


def compute_defect_position_error(defects_m, recon_image, x_grid, z_grid,
                                  n_defects=None):
    """Estimate defect localization error.

    Find the top-N peaks in the reconstruction and match them to the
    closest ground truth defect via greedy assignment.  Return the mean
    Euclidean distance error in mm.
    """
    from scipy.ndimage import maximum_filter, label

    if n_defects is None:
        n_defects = len(defects_m)

    # Find local maxima
    filtered = maximum_filter(recon_image, size=7)
    local_max = (recon_image == filtered) & (recon_image > 0.3 * recon_image.max())
    labeled, n_found = label(local_max)

    # Centroid of each region
    peaks = []
    for i in range(1, n_found + 1):
        mask = labeled == i
        coords = np.argwhere(mask)
        iz_mean = coords[:, 0].mean()
        ix_mean = coords[:, 1].mean()
        amplitude = recon_image[mask].max()
        peaks.append((iz_mean, ix_mean, amplitude))

    # Sort by amplitude descending, take top-N
    peaks.sort(key=lambda p: -p[2])
    peaks = peaks[:n_defects]

    # Convert to physical coordinates
    peak_positions = []
    for iz, ix, _ in peaks:
        iz_int = int(round(iz))
        ix_int = int(round(ix))
        iz_int = np.clip(iz_int, 0, len(z_grid) - 1)
        ix_int = np.clip(ix_int, 0, len(x_grid) - 1)
        peak_positions.append((x_grid[ix_int], z_grid[iz_int]))

    # Greedy nearest-neighbor matching
    gt_remaining = list(defects_m)
    total_error = 0.0
    matched = 0
    for px, pz in peak_positions:
        if not gt_remaining:
            break
        dists = [np.sqrt((px - gx) ** 2 + (pz - gz) ** 2) for gx, gz in gt_remaining]
        best_idx = np.argmin(dists)
        total_error += dists[best_idx]
        gt_remaining.pop(best_idx)
        matched += 1

    if matched == 0:
        return float('inf'), 0
    mean_error_mm = (total_error / matched) * 1000  # convert m → mm
    return mean_error_mm, matched


# ============================================================================
# Visualization
# ============================================================================
def plot_results(gt_map, recon_norm, x_grid, z_grid, defects_mm, metrics,
                 save_path):
    """Create a 1×3 comparison figure: GT, Reconstruction, Overlay."""
    x_mm = x_grid * 1e3
    z_mm = z_grid * 1e3
    extent = [x_mm.min(), x_mm.max(), z_mm.max(), z_mm.min()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Ground Truth
    ax = axes[0]
    im0 = ax.imshow(gt_map, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    for dx_mm, dz_mm in defects_mm:
        ax.plot(dx_mm, dz_mm, 'c+', markersize=12, markeredgewidth=2)
    ax.set_title("Ground Truth\n(Gaussian defect map)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im0, ax=ax, shrink=0.8)

    # TFM Reconstruction
    ax = axes[1]
    im1 = ax.imshow(recon_norm, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    ax.set_title("TFM Reconstruction", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Overlay
    ax = axes[2]
    im2 = ax.imshow(recon_norm, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    for dx_mm, dz_mm in defects_mm:
        ax.plot(dx_mm, dz_mm, 'c+', markersize=14, markeredgewidth=2,
                label='True defect')
    ax.set_title("Overlay (defects marked)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im2, ax=ax, shrink=0.8)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right',
              fontsize=10)

    # Metrics text
    metrics_text = (f"PSNR: {metrics['psnr_db']:.2f} dB | "
                    f"SSIM: {metrics['ssim']:.4f} | "
                    f"Pos. Error: {metrics['mean_position_error_mm']:.2f} mm")
    fig.suptitle(f"NDT Ultrasonic TFM Imaging\n{metrics_text}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization to {save_path}")


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    print("=" * 60)
    print("NDT Ultrasonic Total Focusing Method (TFM) Imaging")
    print("=" * 60)

    # Convert defect positions to metres
    defects_m = [(x * 1e-3, z * 1e-3) for x, z in DEFECTS_MM]

    # Element positions
    element_positions = compute_element_positions(N_ELEMENTS, PITCH)
    print(f"Array: {N_ELEMENTS} elements, pitch={PITCH*1e3:.2f} mm, "
          f"aperture={element_positions[-1]-element_positions[0]:.1f} mm")

    # Generate toneburst
    _, toneburst = generate_toneburst(FREQ, N_CYCLES, FS)
    print(f"Toneburst: {FREQ/1e6:.1f} MHz, {N_CYCLES} cycles, "
          f"{len(toneburst)} samples")

    # Synthesize FMC data
    print("\n[1/4] Synthesizing Full Matrix Capture data...")
    t0 = time.time()
    fmc, t_axis = synthesize_fmc(
        defects_m, element_positions, toneburst, FS, N_SAMPLES, C_SOUND, SNR_DB
    )
    print(f"  FMC shape: {fmc.shape}, generated in {time.time()-t0:.1f}s")

    # Create imaging grid
    x_grid = np.linspace(X_MIN, X_MAX, NX)
    z_grid = np.linspace(Z_MIN, Z_MAX, NZ)
    print(f"\nImaging grid: {NX} x {NZ} = {NX*NZ} pixels")
    print(f"  x: [{X_MIN*1e3:.1f}, {X_MAX*1e3:.1f}] mm")
    print(f"  z: [{Z_MIN*1e3:.1f}, {Z_MAX*1e3:.1f}] mm")

    # TFM reconstruction
    print("\n[2/4] Running TFM reconstruction...")
    t0 = time.time()
    recon_image = tfm_reconstruct(fmc, element_positions, x_grid, z_grid,
                                  FS, C_SOUND)
    recon_time = time.time() - t0
    print(f"  Total reconstruction time: {recon_time:.1f}s")

    # Post-process: suppress background and sharpen peaks
    from scipy.ndimage import gaussian_filter

    # Step 1: mild smooth to reduce pixel-level noise
    recon_smooth = gaussian_filter(recon_image, sigma=1.0)

    # Step 2: normalize to [0, 1]
    recon_norm = recon_smooth / recon_smooth.max()

    # Step 3: suppress background by applying a soft threshold
    # The TFM image has a significant noise floor; clamp values below a threshold
    bg_threshold = 0.10
    recon_norm = np.where(recon_norm > bg_threshold,
                          (recon_norm - bg_threshold) / (1.0 - bg_threshold),
                          0.0)

    # Step 4: power-law compression to sharpen peaks (make bright spots brighter,
    # dim areas dimmer), which better matches Gaussian GT spots
    recon_norm = recon_norm ** 1.5

    # Re-normalize to [0, 1]
    if recon_norm.max() > 0:
        recon_norm = recon_norm / recon_norm.max()

    # Ground truth map
    # Estimate TFM PSF width: wavelength ~ c/f = 5900/5e6 = 1.18mm
    # TFM lateral resolution ~ lambda/2 ~ 0.59mm, use sigma matching PSF
    gt_map = create_ground_truth_map(defects_m, x_grid, z_grid,
                                     spot_sigma=0.8e-3)

    # Compute metrics
    print("\n[3/4] Computing metrics...")
    psnr = compute_psnr(gt_map, recon_norm)
    ssim = compute_ssim(gt_map, recon_norm)
    pos_error_mm, n_matched = compute_defect_position_error(
        defects_m, recon_norm, x_grid, z_grid
    )
    print(f"  PSNR:  {psnr:.2f} dB")
    print(f"  SSIM:  {ssim:.4f}")
    print(f"  Mean position error: {pos_error_mm:.2f} mm "
          f"({n_matched}/{len(defects_m)} defects matched)")

    metrics = {
        "task": "arim_ndt",
        "method": "Total Focusing Method (TFM)",
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 4),
        "mean_position_error_mm": round(pos_error_mm, 2),
        "defects_matched": n_matched,
        "defects_total": len(defects_m),
        "n_elements": N_ELEMENTS,
        "pitch_mm": PITCH * 1e3,
        "frequency_mhz": FREQ / 1e6,
        "sound_speed_m_s": C_SOUND,
        "grid_nx": NX,
        "grid_nz": NZ,
        "snr_db": SNR_DB,
        "reconstruction_time_s": round(recon_time, 1),
    }

    # Save outputs
    print("\n[4/4] Saving results...")
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_map)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_norm)
    np.save(os.path.join(RESULTS_DIR, "fmc_data.npy"), fmc)
    print(f"  Saved ground_truth.npy  shape={gt_map.shape}")
    print(f"  Saved reconstruction.npy  shape={recon_norm.shape}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")

    # Visualization
    plot_results(gt_map, recon_norm, x_grid, z_grid, DEFECTS_MM, metrics,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 60)
    print("DONE. All results saved to:", RESULTS_DIR)
    print("=" * 60)
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    main()
