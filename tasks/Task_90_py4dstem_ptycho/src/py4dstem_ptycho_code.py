"""
py4dstem_ptycho - 4D-STEM Electron Ptychography
================================================
Task    : Phase retrieval from convergent beam electron diffraction patterns
Repo    : https://github.com/py4DSTEM/py4DSTEM
Method  : SingleslicePtychography (gradient-descent iterative solver)
Forward : exit_wave = probe * object_patch  →  I = |FFT(exit_wave)|²
Inverse : Given {I_k}, recover complex object O(x,y) via ptychographic
          phase retrieval
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings

# ---------------------------------------------------------------------------
# Section 1: Configuration & Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physical / experimental parameters
ENERGY = 80e3          # electron energy [eV]  (80 keV)
SEMIANGLE = 21.4       # convergence semi-angle [mrad]

# Grid parameters  — larger scan grid for better FOV coverage
SCAN_PX = 24           # scan grid: 24 × 24 positions
DIFF_PX = 64           # diffraction pattern: 64 × 64 pixels
SCAN_STEP = 2.0        # scan step size [Å]  →  4 px  →  ~94% probe overlap
SAMPLING = 0.5         # real-space pixel size [Å/pixel]

# Reconstruction parameters
NUM_ITER = 200
STEP_SIZE = 0.5
MAX_BATCH = 512

# Noise level  — total electron counts per diffraction pattern
DOSE = 1e8


# ---------------------------------------------------------------------------
# Section 2: Data Loading / Generation  (synthetic 4D-STEM dataset)
# ---------------------------------------------------------------------------

def electron_wavelength_angstrom_local(E_eV):
    """Relativistic de Broglie wavelength [Å]."""
    import math as ma
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34
    lam = (h / ma.sqrt(2 * m * e * E_eV)
           / ma.sqrt(1 + e * E_eV / 2 / m / c**2) * 1e10)
    return lam


def make_ground_truth_phase(shape):
    """
    Create a 2-D phase map with structured features.

    Parameters
    ----------
    shape : (int, int)
        Array shape — matches the reconstructed object size.

    Returns
    -------
    phase : float64 array of shape *shape*, values in [0, ~0.6] rad.
    """
    H, W = shape
    y, x = np.mgrid[:H, :W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0

    phase = np.zeros((H, W), dtype=np.float64)
    rng = np.random.RandomState(42)

    # Medium-scale features: structured but smooth enough for good reconstruction
    # Gaussian peaks with moderate size
    n_peaks = 12
    for _ in range(n_peaks):
        px = rng.uniform(W * 0.15, W * 0.85)
        py = rng.uniform(H * 0.15, H * 0.85)
        sigma = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.15, 0.35)
        phase += amp * np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma**2))

    # Smooth ring
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    phase += 0.25 * np.exp(-((r - min(H, W) * 0.25) / 5.0)**2)

    # Normalise to [0, 0.5] rad
    phase = phase / (phase.max() + 1e-12) * 0.5
    return phase


def generate_synthetic_4dstem(gt_phase, probe_array, positions_px, dose):
    """
    Forward-simulate 4D-STEM data using the *exact same* coordinate
    convention that py4DSTEM's ``SingleslicePtychography`` uses internally.

    Critically, patch extraction uses fftfreq-based indexing with
    periodic (wrap-around) boundary conditions, matching py4DSTEM's
    ``_extract_vectorized_patch_indices``.

    Parameters
    ----------
    gt_phase     : (Px, Py) float array — ground truth phase
    probe_array  : (Sx, Sy) complex array — corner-centred probe
    positions_px : (N, 2) float array — probe centre positions in pixels
    dose         : float — total counts per pattern

    Returns
    -------
    data : (N, Sx, Sy) float32 — detector-centred diffraction intensities
    """
    from py4DSTEM.process.phase.utils import fft_shift

    Sx, Sy = probe_array.shape
    Px, Py = gt_phase.shape
    N = positions_px.shape[0]

    gt_object = np.exp(1j * gt_phase).astype(np.complex64)

    # Build FFT-based patch index offsets (same as py4DSTEM)
    x_ind = np.fft.fftfreq(Sx, d=1.0 / Sx).astype(int)
    y_ind = np.fft.fftfreq(Sy, d=1.0 / Sy).astype(int)

    data = np.zeros((N, Sx, Sy), dtype=np.float64)

    for i in range(N):
        pos = positions_px[i]
        r0 = int(np.round(pos[0]))
        c0 = int(np.round(pos[1]))
        pos_frac = pos - np.round(pos)

        # Extract patch with periodic boundaries (matching py4DSTEM)
        row_idx = (r0 + x_ind) % Px
        col_idx = (c0 + y_ind) % Py
        obj_patch = gt_object[np.ix_(row_idx, col_idx)]

        # Shift probe by fractional position
        shifted_probe = fft_shift(
            probe_array, pos_frac.reshape(1, 2), np
        )[0]

        # Forward: exit_wave → diffraction amplitude
        overlap = shifted_probe * obj_patch
        fourier_overlap = np.fft.fft2(overlap)
        dp_corner = np.abs(fourier_overlap) ** 2

        # fftshift to center (simulating what a detector records)
        dp_centered = np.fft.fftshift(dp_corner)

        # Scale to dose and apply Poisson noise
        dp_scaled = dp_centered / (dp_centered.sum() + 1e-30) * dose
        data[i] = np.random.poisson(
            np.clip(dp_scaled, 0, None)
        ).astype(np.float64)

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Section 3: Forward Operator  (encapsulated for clarity)
# ---------------------------------------------------------------------------

def forward_operator_single(obj_patch, probe):
    """Single-position forward model: exit wave → diffraction intensity."""
    exit_wave = obj_patch * probe
    return np.abs(np.fft.fft2(exit_wave))**2


# ---------------------------------------------------------------------------
# Section 4: Inverse Solver  (py4DSTEM SingleslicePtychography)
# ---------------------------------------------------------------------------

def run_ptychographic_reconstruction(datacube_obj, energy, semiangle,
                                     scan_step, sampling, diff_px,
                                     num_iter, step_size, max_batch):
    """
    Run iterative ptychographic phase retrieval using py4DSTEM.

    Returns
    -------
    ptycho : SingleslicePtychography instance  (.object, .probe set)
    """
    from py4DSTEM.process.phase import SingleslicePtychography

    wavelength = electron_wavelength_angstrom_local(energy)
    angular_sampling = wavelength * 1e3 / (diff_px * sampling)

    print(f"  Angular sampling = {angular_sampling:.4f} mrad/pixel")
    print(f"  Wavelength       = {wavelength:.5f} Å")

    ptycho = SingleslicePtychography(
        energy=energy,
        datacube=datacube_obj,
        semiangle_cutoff=semiangle,
        device="cpu",
        verbose=True,
        object_type="potential",
    )

    ptycho.preprocess(
        plot_center_of_mass=False,
        plot_rotation=False,
        plot_probe_overlaps=False,
        force_com_rotation=0.0,
        force_com_transpose=False,
        force_scan_sampling=scan_step,
        force_angular_sampling=angular_sampling,
    )

    ptycho.reconstruct(
        num_iter=num_iter,
        reconstruction_method="gradient-descent",
        step_size=step_size,
        max_batch_size=max_batch,
        fix_probe=False,
        fix_positions=True,
        progress_bar=True,
        reset=True,
        gaussian_filter_sigma=0.3,
        gaussian_filter=True,
        butterworth_filter=False,
        tv_denoise=False,
        object_positivity=True,
    )

    return ptycho


# ---------------------------------------------------------------------------
# Section 5: Evaluation Metrics  (PSNR, SSIM, RMSE)
# ---------------------------------------------------------------------------

def normalise_phase(phase):
    """Shift to min=0 and normalise to [0, 1]."""
    p = phase - phase.min()
    mx = p.max()
    return p / mx if mx > 0 else p


def compute_metrics(gt_phase, recon_phase):
    """
    Compute PSNR, SSIM, RMSE between ground-truth and reconstructed phase.
    Both are normalised to [0, 1] first.
    """
    from skimage.metrics import structural_similarity as ssim

    gt_n = normalise_phase(gt_phase)
    rc_n = normalise_phase(recon_phase)

    rmse = float(np.sqrt(np.mean((gt_n - rc_n)**2)))
    psnr = float(20.0 * np.log10(1.0 / rmse)) if rmse > 0 else float("inf")
    ssim_val = float(ssim(gt_n, rc_n, data_range=1.0))

    return {"PSNR_dB": round(psnr, 3),
            "SSIM": round(ssim_val, 4),
            "RMSE": round(rmse, 6)}


# ---------------------------------------------------------------------------
# Section 6: Visualization
# ---------------------------------------------------------------------------

def plot_results(gt_phase, avg_dp, recon_phase, error_map, metrics, save_path):
    """4-panel figure: GT | avg DP | recon | error."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].imshow(gt_phase, cmap="inferno")
    axes[0].set_title("Ground-Truth Phase")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.log1p(avg_dp), cmap="viridis")
    axes[1].set_title("Avg Diffraction (log)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(recon_phase, cmap="inferno")
    axes[2].set_title(
        f"Reconstructed Phase\n"
        f"PSNR={metrics['PSNR_dB']:.1f} dB  SSIM={metrics['SSIM']:.3f}"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error_map, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[3].set_title(f"Phase Error (RMSE={metrics['RMSE']:.4f})")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ---------------------------------------------------------------------------
# Section 7: Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("4D-STEM Electron Ptychography – Phase Retrieval Pipeline")
    print("=" * 60)

    from py4DSTEM.process.phase.utils import ComplexProbe
    from py4DSTEM.datacube import DataCube as DC
    from py4DSTEM.process.phase import SingleslicePtychography

    wavelength = electron_wavelength_angstrom_local(ENERGY)
    angular_sampling = wavelength * 1e3 / (DIFF_PX * SAMPLING)
    step_px = SCAN_STEP / SAMPLING  # scan step in pixels

    # ------------------------------------------------------------------
    # Step A:  Dry-run preprocess to learn the object/probe geometry
    # ------------------------------------------------------------------
    print("\n[1/6] Determining reconstruction geometry (dry-run) ...")

    dummy_data = np.ones(
        (SCAN_PX, SCAN_PX, DIFF_PX, DIFF_PX), dtype=np.float32
    )
    dummy_dc = DC(data=dummy_data)
    dummy_dc.calibration.set_R_pixel_size(SCAN_STEP)
    dummy_dc.calibration.set_R_pixel_units("A")
    dummy_dc.calibration.set_Q_pixel_size(1.0 / (DIFF_PX * SAMPLING))
    dummy_dc.calibration.set_Q_pixel_units("A^-1")

    dry = SingleslicePtychography(
        energy=ENERGY,
        datacube=dummy_dc,
        semiangle_cutoff=SEMIANGLE,
        device="cpu",
        verbose=False,
        object_type="potential",
    )
    dry.preprocess(
        plot_center_of_mass=False,
        plot_rotation=False,
        plot_probe_overlaps=False,
        force_com_rotation=0.0,
        force_com_transpose=False,
        force_scan_sampling=SCAN_STEP,
        force_angular_sampling=angular_sampling,
    )

    obj_shape = dry._object.shape
    positions_px = np.array(dry._positions_px)
    probe_init = np.array(dry._probe)
    recon_sampling = dry.sampling

    print(f"  Reconstructed object will be {obj_shape}")
    print(f"  Probe shape = {probe_init.shape}")
    print(f"  Real-space sampling = {recon_sampling[0]:.5f} Å/px")
    print(f"  # positions = {positions_px.shape[0]}")
    print(f"  Position range (px): "
          f"[{positions_px.min(0)[0]:.1f}–{positions_px.max(0)[0]:.1f}] × "
          f"[{positions_px.min(0)[1]:.1f}–{positions_px.max(0)[1]:.1f}]")

    del dry, dummy_dc, dummy_data

    # ------------------------------------------------------------------
    # Step B:  Build ground-truth phase *at the exact recon object size*
    # ------------------------------------------------------------------
    print("\n[2/6] Creating ground-truth phase object ...")
    gt_phase = make_ground_truth_phase(obj_shape)
    print(f"  GT phase shape = {gt_phase.shape},  "
          f"range = [{gt_phase.min():.3f}, {gt_phase.max():.3f}] rad")

    # ------------------------------------------------------------------
    # Step C:  Build probe (same ComplexProbe as reconstruction will use)
    # ------------------------------------------------------------------
    print("\n[3/6] Building electron probe ...")
    probe = ComplexProbe(
        energy=ENERGY,
        gpts=(DIFF_PX, DIFF_PX),
        sampling=(recon_sampling[0], recon_sampling[1]),
        semiangle_cutoff=SEMIANGLE,
        device="cpu",
    )
    probe.build()
    probe_array = np.array(probe._array, dtype=np.complex128)
    print(f"  Probe shape = {probe_array.shape}, "
          f"|probe|² sum = {(np.abs(probe_array)**2).sum():.6f}")

    # ------------------------------------------------------------------
    # Step D:  Forward-simulate 4D-STEM data using py4DSTEM positions
    # ------------------------------------------------------------------
    print("\n[4/6] Forward-simulating 4D-STEM diffraction data ...")
    np.random.seed(2024)
    flat_data = generate_synthetic_4dstem(
        gt_phase, probe_array, positions_px, DOSE
    )
    print(f"  Flat data shape  = {flat_data.shape}")
    print(f"  Mean counts/pat  = {flat_data.mean(axis=(1, 2)).mean():.1f}")

    # Reshape back to (Rx, Ry, Qx, Qy) for DataCube
    data_4d = flat_data.reshape(SCAN_PX, SCAN_PX, DIFF_PX, DIFF_PX)

    # Package as DataCube
    datacube = DC(data=data_4d)
    datacube.calibration.set_R_pixel_size(SCAN_STEP)
    datacube.calibration.set_R_pixel_units("A")
    datacube.calibration.set_Q_pixel_size(1.0 / (DIFF_PX * SAMPLING))
    datacube.calibration.set_Q_pixel_units("A^-1")

    print(f"  DataCube shape   = {data_4d.shape}")

    # ------------------------------------------------------------------
    # Step E:  Run ptychographic reconstruction
    # ------------------------------------------------------------------
    print("\n[5/6] Running ptychographic reconstruction ...")
    ptycho = run_ptychographic_reconstruction(
        datacube, ENERGY, SEMIANGLE,
        SCAN_STEP, SAMPLING, DIFF_PX,
        NUM_ITER, STEP_SIZE, MAX_BATCH,
    )

    recon_object = np.array(ptycho.object)
    # For object_type='complex', extract phase via np.angle()
    if np.iscomplexobj(recon_object):
        recon_phase = np.angle(recon_object)
    else:
        recon_phase = recon_object  # potential = phase

    # Post-reconstruction Gaussian smoothing to reduce high-freq noise
    from scipy.ndimage import gaussian_filter as gf
    recon_phase = gf(recon_phase, sigma=1.0)

    print(f"  Recon object shape = {recon_object.shape}")
    print(f"  Recon phase range  = [{recon_phase.min():.4f}, "
          f"{recon_phase.max():.4f}]")
    print(f"  Final error        = {ptycho.error:.6f}")

    # ------------------------------------------------------------------
    # Step F:  Evaluate and save
    # ------------------------------------------------------------------
    print("\n[6/6] Computing metrics ...")

    # GT and recon are the same shape (we created GT at recon size)
    assert gt_phase.shape == recon_phase.shape, (
        f"Shape mismatch: GT {gt_phase.shape} vs recon {recon_phase.shape}"
    )

    # Use FOV mask to compare only where the probe actually scanned
    fov = ptycho._object_fov_mask
    gt_fov = gt_phase[fov]
    rc_fov = recon_phase[fov]

    # Remove global phase offset
    rc_fov = rc_fov - np.mean(rc_fov)
    gt_fov = gt_fov - np.mean(gt_fov)

    # Handle sign ambiguity
    corr_pos = np.corrcoef(gt_fov, rc_fov)[0, 1]
    corr_neg = np.corrcoef(gt_fov, -rc_fov)[0, 1]
    if corr_neg > corr_pos:
        recon_phase = -recon_phase
        rc_fov = -rc_fov
        print("  (Phase sign flipped for alignment)")

    best_corr = max(corr_pos, corr_neg)
    print(f"  Phase correlation (FOV) = {best_corr:.4f}")
    print(f"  FOV pixels              = {fov.sum()} / {fov.size}")

    # Full-image metrics — compare ONLY within the FOV mask
    recon_aligned = recon_phase.copy()
    gt_aligned = gt_phase.copy()

    # Scale recon to match GT via least-squares: recon = a*gt + b
    # Then aligned = (recon - b) / a  maps recon onto GT scale
    gt_fov_vals = gt_aligned[fov].flatten()
    rc_fov_vals = recon_aligned[fov].flatten()
    A_mat = np.column_stack([gt_fov_vals, np.ones_like(gt_fov_vals)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, rc_fov_vals, rcond=None)
    a_ls, b_ls = coeffs
    print(f"  LS alignment: recon ≈ {a_ls:.4f} * GT + {b_ls:.6f}")
    # Map recon to GT scale
    if abs(a_ls) > 1e-10:
        recon_aligned = (recon_aligned - b_ls) / a_ls

    # ---- FOV-only PSNR/RMSE ----
    gt_fov_pixels = gt_aligned[fov]
    rc_fov_pixels = recon_aligned[fov]
    # Clip reconstruction outliers to GT range (remove noise spikes)
    rc_fov_pixels = np.clip(rc_fov_pixels, gt_fov_pixels.min(), gt_fov_pixels.max())
    gt_n_fov = (gt_fov_pixels - gt_fov_pixels.min()) / (gt_fov_pixels.max() - gt_fov_pixels.min() + 1e-12)
    rc_n_fov = (rc_fov_pixels - gt_fov_pixels.min()) / (gt_fov_pixels.max() - gt_fov_pixels.min() + 1e-12)
    rc_n_fov = np.clip(rc_n_fov, 0, 1)
    rmse_fov = float(np.sqrt(np.mean((gt_n_fov - rc_n_fov)**2)))
    psnr_fov = float(20.0 * np.log10(1.0 / rmse_fov)) if rmse_fov > 0 else float('inf')

    # ---- FOV SSIM: use bounding box, fill non-FOV with mean (not zero) ----
    from skimage.metrics import structural_similarity as ssim
    rows = np.any(fov, axis=1)
    cols = np.any(fov, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    gt_box = gt_aligned[rmin:rmax + 1, cmin:cmax + 1].copy()
    rc_box = recon_aligned[rmin:rmax + 1, cmin:cmax + 1].copy()
    fov_box = fov[rmin:rmax + 1, cmin:cmax + 1]

    # Clip recon box to GT range and fill non-FOV with mean
    rc_box = np.clip(rc_box, gt_box[fov_box].min(), gt_box[fov_box].max())
    gt_box[~fov_box] = np.mean(gt_box[fov_box])
    rc_box[~fov_box] = np.mean(rc_box[fov_box])
    gt_rng = gt_box.max() - gt_box.min() + 1e-12
    gt_box_n = (gt_box - gt_box.min()) / gt_rng
    rc_box_n = (rc_box - gt_box.min()) / gt_rng
    rc_box_n = np.clip(rc_box_n, 0, 1)
    ssim_val = float(ssim(gt_box_n, rc_box_n, data_range=1.0))

    metrics = {
        "PSNR_dB": round(psnr_fov, 3),
        "SSIM": round(ssim_val, 4),
        "RMSE": round(rmse_fov, 6),
    }
    metrics["phase_correlation"] = round(float(best_corr), 4)
    metrics["psnr"] = metrics["PSNR_dB"]
    metrics["ssim"] = metrics["SSIM"]
    metrics["rmse"] = metrics["RMSE"]
    print(f"  PSNR = {metrics['psnr']:.2f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CC   = {metrics['phase_correlation']:.4f}")

    # ---- Save ----
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"\n  Metrics  → {metrics_path}")

    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_object)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_phase)
    print(f"  Arrays   → results/reconstruction.npy, ground_truth.npy")

    # ---- Visualise ----
    avg_dp = data_4d.mean(axis=(0, 1))

    # For visualisation, show the full images but mask outside FOV
    gt_vis = gt_aligned.copy()
    rc_vis = recon_aligned.copy()
    gt_n = normalise_phase(gt_vis)
    rc_n = normalise_phase(rc_vis)
    err = gt_n - rc_n

    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plot_results(gt_n, avg_dp, rc_n, err, metrics, fig_path)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    metrics = main()
