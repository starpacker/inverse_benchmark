"""
GPRPy — Ground Penetrating Radar Depth Migration
==================================================
Task #68: Reconstruct subsurface reflectivity from GPR B-scan data
          using Kirchhoff depth migration.

Inverse Problem:
    Given a GPR B-scan d(x,t) (trace ensemble), recover the subsurface
    reflectivity image r(x,z) by migrating diffractions back to their
    true spatial locations.

Forward Model:
    Exploding reflector model:
    d(x,t) = ∫∫ r(x',z) · w(t - 2√((x-x')²+z²)/v) dx' dz
    where v is the EM wave velocity and w(t) is the source wavelet.

Inverse Solver:
    Kirchhoff depth migration — time-domain summation along hyperbolic
    travel-time curves:
    r̂(x,z) = Σ_x' d(x', t=2R/v) · cos(θ)/R

Repo: https://github.com/NSGeophysics/GPRPy
Paper: Annan (2005), Near-Surface Geophysics, SEG.

Usage: /data/yjh/spectro_env/bin/python GPRPy_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NX_TRACES = 100            # Number of GPR traces (survey positions)
NT = 512                   # Samples per trace
DX = 0.05                  # Trace spacing [m]
DT = 0.1e-9                # Time sampling [s] (0.1 ns)
V_EM = 0.1                 # EM velocity [m/ns] ≈ εr=9
FREQ_CENTER = 400e6        # Centre frequency [Hz]
NOISE_SNR_DB = 35          # SNR [dB]
SEED = 42

# Depth extent
Z_MAX = V_EM * 1e9 * NT * DT / 2  # Max depth [m]
NZ = NT // 2                       # Depth samples


# ─── Data Generation ──────────────────────────────────────────────
def generate_subsurface_model(nx, nz, dx, z_max):
    """
    Create a synthetic subsurface reflectivity model with:
    - Horizontal layers
    - Dipping interface
    - Point diffractors (pipes, boulders)
    - Void (tunnel)
    """
    z = np.linspace(0, z_max, nz)
    x = np.arange(nx) * dx
    reflectivity = np.zeros((nx, nz))

    # Layer 1: horizontal at z = 0.5 m
    iz1 = np.argmin(np.abs(z - 0.5))
    reflectivity[:, iz1] = 0.3

    # Layer 2: dipping interface
    for ix in range(nx):
        z_dip = 1.0 + 0.3 * (ix / nx)
        iz_dip = np.argmin(np.abs(z - z_dip))
        if iz_dip < nz:
            reflectivity[ix, iz_dip] = 0.5

    # Layer 3: horizontal at z = 1.8 m
    iz3 = np.argmin(np.abs(z - 1.8))
    reflectivity[:, iz3] = 0.4

    # Point diffractors
    diffractors = [
        (nx // 4, 0.8, 0.7),      # Pipe at shallow depth
        (nx // 2, 1.3, 0.6),      # Boulder
        (3 * nx // 4, 0.6, 0.8),  # Rebar
    ]
    for ix, z_d, amp in diffractors:
        iz_d = np.argmin(np.abs(z - z_d))
        if 0 <= ix < nx and 0 <= iz_d < nz:
            reflectivity[ix, iz_d] = amp

    # Small void (rectangular)
    ix_void_start = int(0.6 * nx)
    ix_void_end = int(0.65 * nx)
    iz_void = np.argmin(np.abs(z - 1.1))
    iz_void_end = np.argmin(np.abs(z - 1.3))
    reflectivity[ix_void_start:ix_void_end, iz_void] = 0.6
    reflectivity[ix_void_start:ix_void_end, iz_void_end] = 0.6
    reflectivity[ix_void_start, iz_void:iz_void_end] = 0.5
    reflectivity[ix_void_end, iz_void:iz_void_end] = 0.5

    return reflectivity, x, z


def generate_wavelet(freq, dt, n_samples=64):
    """Generate Ricker (Mexican hat) wavelet for GPR source."""
    t = np.arange(n_samples) * dt
    t_centre = t[n_samples // 2]
    tau = t - t_centre
    sigma = 1.0 / (np.pi * freq * np.sqrt(2))
    wavelet = (1 - (tau / sigma)**2) * np.exp(-0.5 * (tau / sigma)**2)
    wavelet /= np.max(np.abs(wavelet))
    return wavelet


def forward_exploding_reflector(reflectivity, x_traces, z, dt, nt, v, wavelet, rng):
    """
    Exploding reflector model for GPR B-scan synthesis (vectorised).
    Each reflector explodes simultaneously, waves recorded at surface.
    """
    nx = len(x_traces)
    bscan_clean = np.zeros((nx, nt))
    dx_scene = x_traces[1] - x_traces[0] if len(x_traces) > 1 else 0.05
    n_wav = len(wavelet)

    # Get non-zero reflector positions
    nz_idx = np.argwhere(reflectivity > 1e-10)
    if len(nz_idx) == 0:
        noise = rng.standard_normal(bscan_clean.shape) * 1e-6
        return bscan_clean, bscan_clean + noise

    r_vals = reflectivity[nz_idx[:, 0], nz_idx[:, 1]]  # (K,)
    x_refl = nz_idx[:, 0].astype(float) * dx_scene       # (K,)
    z_refl = z[nz_idx[:, 1]]                               # (K,)

    print(f"  Generating B-scan ({nx} traces, {len(r_vals)} reflectors) ...")

    for ix_t in range(nx):
        x_recv = x_traces[ix_t]
        # Vectorised over all reflectors
        dist = np.sqrt((x_recv - x_refl)**2 + z_refl**2)  # (K,)
        twt = 2 * dist / (v * 1e9)
        it_arr = (twt / dt).astype(int)  # (K,)

        valid = (it_arr >= 0) & (it_arr < nt)
        for k in np.where(valid)[0]:
            it = it_arr[k]
            it_start = max(0, it - n_wav // 2)
            it_end = min(nt, it + n_wav // 2)
            wav_start = it_start - (it - n_wav // 2)
            wav_end = wav_start + (it_end - it_start)
            bscan_clean[ix_t, it_start:it_end] += r_vals[k] * wavelet[wav_start:wav_end]

    # Add noise
    sig_power = np.mean(bscan_clean**2)
    noise_power = sig_power / (10**(NOISE_SNR_DB / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(bscan_clean.shape)
    bscan_noisy = bscan_clean + noise

    return bscan_clean, bscan_noisy


# ─── Inverse Solver: Kirchhoff Migration ──────────────────────────
def kirchhoff_migration(bscan, x_traces, z_img, dt, v):
    """
    Kirchhoff depth migration for GPR data (vectorised over traces).

    For each image point (x, z):
    r̂(x,z) = Σ_{x'} d(x', t=2R/v) · cos(θ) / R
    """
    nx = len(x_traces)
    nz = len(z_img)
    nt = bscan.shape[1]
    image = np.zeros((nx, nz))

    print(f"  Migrating {nx} traces × {nz} depth samples ...")

    x_tr_arr = np.asarray(x_traces)  # (nx,)

    for ix_img in range(nx):
        x_pt = x_traces[ix_img]
        for iz_img in range(nz):
            z_pt = z_img[iz_img]
            if z_pt < 1e-6:
                continue

            # Vectorised over all traces
            R = np.sqrt((x_pt - x_tr_arr)**2 + z_pt**2)  # (nx,)
            twt = 2 * R / (v * 1e9)
            it_float = twt / dt  # (nx,)
            it_int = np.floor(it_float).astype(int)
            frac = it_float - it_int

            valid = (it_int >= 0) & (it_int < nt - 1)
            it_safe = np.clip(it_int, 0, nt - 2)

            d_interp = (1 - frac) * bscan[np.arange(nx), it_safe] + \
                       frac * bscan[np.arange(nx), it_safe + 1]

            cos_theta = z_pt / np.maximum(R, 1e-10)
            weight = cos_theta / np.maximum(R, 1e-6)

            image[ix_img, iz_img] = np.sum(valid * d_interp * weight)

    return image


def fk_migration(bscan, dx, dt, v):
    """
    FK (Stolt) migration — frequency-wavenumber domain.
    Faster alternative to Kirchhoff for constant velocity.
    """
    nx, nt = bscan.shape

    # 2D FFT
    D = fft2(bscan)

    # Wavenumber and frequency axes
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    omega = np.fft.fftfreq(nt, d=dt) * 2 * np.pi

    # Stolt mapping: ω → kz
    v_ms = v * 1e9  # m/ns to m/s
    image_fk = np.zeros((nx, nt // 2), dtype=complex)

    for ikx in range(nx):
        for iw in range(nt):
            w = omega[iw]
            k = kx[ikx]

            # Dispersion relation: kz = √((2ω/v)² - kx²)
            arg = (2 * w / v_ms)**2 - k**2
            if arg > 0:
                kz = np.sqrt(arg)
                iz = int(kz / (2 * np.pi / (nt * dt * v_ms / 2)) * (nt // 2))
                if 0 <= iz < nt // 2:
                    image_fk[ikx, iz] += D[ikx, iw]

    return np.abs(ifft2(image_fk, s=[nx, nt // 2]))


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt, rec):
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(reflectivity, bscan, migrated, x, z, t_axis, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Subsurface model
    axes[0, 0].imshow(reflectivity.T, aspect='auto', cmap='gray_r',
                       extent=[x[0], x[-1], z[-1], z[0]])
    axes[0, 0].set_title('True Reflectivity Model')
    axes[0, 0].set_xlabel('Position [m]')
    axes[0, 0].set_ylabel('Depth [m]')

    # B-scan
    clip = np.percentile(np.abs(bscan), 98)
    axes[0, 1].imshow(bscan.T, aspect='auto', cmap='RdBu_r', vmin=-clip, vmax=clip,
                       extent=[x[0], x[-1], t_axis[-1]*1e9, t_axis[0]*1e9])
    axes[0, 1].set_title('GPR B-Scan (noisy)')
    axes[0, 1].set_xlabel('Position [m]')
    axes[0, 1].set_ylabel('Two-way time [ns]')

    # Migrated image
    axes[1, 0].imshow(migrated.T, aspect='auto', cmap='gray_r',
                       extent=[x[0], x[-1], z[-1], z[0]])
    axes[1, 0].set_title('Kirchhoff Migration')
    axes[1, 0].set_xlabel('Position [m]')
    axes[1, 0].set_ylabel('Depth [m]')

    # Cross-section comparison
    mid = reflectivity.shape[0] // 2
    axes[1, 1].plot(z, reflectivity[mid, :] / max(reflectivity[mid, :].max(), 1e-12),
                     'b-', lw=2, label='GT')
    axes[1, 1].plot(z, migrated[mid, :] / max(migrated[mid, :].max(), 1e-12),
                     'r--', lw=2, label='Migrated')
    axes[1, 1].set_title(f'Trace {mid} Comparison')
    axes[1, 1].set_xlabel('Depth [m]')
    axes[1, 1].legend()

    fig.suptitle(
        f"GPRPy — GPR Migration\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  GPRPy — Ground Penetrating Radar Migration")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation")
    reflectivity, x_traces, z_depth = generate_subsurface_model(
        NX_TRACES, NZ, DX, Z_MAX
    )
    print(f"  Model: {reflectivity.shape}, dx={DX} m, z_max={Z_MAX:.2f} m")
    print(f"  Reflectors: {np.count_nonzero(reflectivity)} non-zero cells")

    # Stage 2: Forward — B-scan synthesis
    print("\n[STAGE 2] Forward — Exploding Reflector B-Scan")
    wavelet = generate_wavelet(FREQ_CENTER, DT)
    bscan_clean, bscan_noisy = forward_exploding_reflector(
        reflectivity, x_traces, z_depth, DT, NT, V_EM, wavelet, rng
    )
    t_axis = np.arange(NT) * DT
    print(f"  B-scan: {bscan_noisy.shape}")
    print(f"  Signal range: [{bscan_clean.min():.3f}, {bscan_clean.max():.3f}]")

    # Stage 3: Inverse — Kirchhoff migration
    print("\n[STAGE 3] Inverse — Kirchhoff Depth Migration")
    migrated = kirchhoff_migration(bscan_noisy, x_traces, z_depth, DT, V_EM)
    # Reflectivity is physically non-negative; clip migration artifacts
    migrated = np.clip(migrated, 0, None)
    print(f"  Migrated image: {migrated.shape}")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    metrics = compute_metrics(reflectivity, migrated)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), migrated)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), reflectivity)

    visualize_results(reflectivity, bscan_noisy, migrated, x_traces, z_depth,
                      t_axis, metrics, os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
