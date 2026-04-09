"""
cubical_cal - Radio Interferometry Gain Calibration
====================================================
Inverse Problem: From corrupted visibilities, recover per-antenna complex gains
Method: Stefcal (iterative least-squares) calibration

Forward model: V_ij^obs = g_i * V_ij^model * conj(g_j) + noise
Inverse: Solve for {g_i} given {V_ij^obs} and {V_ij^model}

Simulates a small radio interferometer array (7 antennas, similar to KAT-7),
generates synthetic visibilities from a point-source sky model, corrupts them
with per-antenna complex gains and thermal noise, then recovers the gains
using the Stefcal algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

N_ANT = 7           # Number of antennas
N_FREQ = 16         # Number of frequency channels
N_TIME = 32         # Number of time slots
N_SRC = 3           # Number of point sources in sky model
SNR_DB = 25.0       # Signal-to-noise ratio in dB
MAX_ITER = 300      # Maximum Stefcal iterations
CONV_TOL = 1e-7     # Convergence tolerance (relative change in gains)
REF_ANT = 0         # Reference antenna (gain fixed to 1+0j)

SEED = 42
RNG = np.random.default_rng(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
#  Sky model
# ──────────────────────────────────────────────────────────────────────

def generate_sky_model(n_src: int, n_freq: int) -> tuple:
    """
    Generate a simple point-source sky model.
    Returns source fluxes (n_src, n_freq) and direction cosines (n_src, 2).
    Flux follows a power-law: S(f) = S0 * (f / f0)^alpha
    """
    s0 = RNG.uniform(1.0, 10.0, size=n_src)           # Flux at reference freq
    alpha = RNG.uniform(-1.5, -0.5, size=n_src)        # Spectral index
    freqs = np.linspace(0.9, 1.7, n_freq)              # GHz
    f0 = 1.3                                            # Reference freq GHz
    fluxes = s0[:, None] * (freqs[None, :] / f0) ** alpha[:, None]  # (n_src, n_freq)

    # Direction cosines (l, m) – small values near phase centre
    lm = RNG.uniform(-0.01, 0.01, size=(n_src, 2))
    return fluxes, lm, freqs


# ──────────────────────────────────────────────────────────────────────
#  Antenna layout and baseline UVW
# ──────────────────────────────────────────────────────────────────────

def generate_antenna_layout(n_ant: int) -> np.ndarray:
    """
    Generate antenna positions in metres (East-North-Up).
    Uses a compact random layout within a 1-km diameter.
    """
    positions = RNG.uniform(-500.0, 500.0, size=(n_ant, 3))
    positions[:, 2] = 0.0  # Flat array
    return positions


def compute_baselines(positions: np.ndarray) -> tuple:
    """Compute baseline vectors and antenna-pair indices."""
    n_ant = positions.shape[0]
    ant1, ant2 = [], []
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            ant1.append(i)
            ant2.append(j)
    ant1 = np.array(ant1)
    ant2 = np.array(ant2)
    uvw = positions[ant2] - positions[ant1]  # (n_bl, 3)
    return ant1, ant2, uvw


# ──────────────────────────────────────────────────────────────────────
#  Model visibilities
# ──────────────────────────────────────────────────────────────────────

def compute_model_visibilities(
    fluxes: np.ndarray,
    lm: np.ndarray,
    uvw: np.ndarray,
    freqs: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """
    Compute model visibilities using the RIME (Radio Interferometer
    Measurement Equation) for point sources:
        V_model(u,v) = sum_s  S_s * exp(-2πi (u*l_s + v*m_s) * f / c)

    Returns: (n_bl, n_freq, n_time) complex array
    """
    c = 2.998e8  # speed of light m/s
    n_bl = uvw.shape[0]
    n_freq = freqs.shape[0]
    n_src = fluxes.shape[0]

    # Convert freqs to Hz
    freqs_hz = freqs * 1e9

    # Slow Earth-rotation effect: rotate UVW slightly per time step
    hour_angles = np.linspace(-0.5, 0.5, n_time)  # hours
    omega = 2.0 * np.pi / (24.0 * 3600.0)  # rad/s

    v_model = np.zeros((n_bl, n_freq, n_time), dtype=np.complex128)

    for t_idx, ha in enumerate(hour_angles):
        rot_angle = omega * ha * 3600.0  # rad
        cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
        # Rotate u, v
        u_rot = uvw[:, 0] * cos_r - uvw[:, 1] * sin_r   # (n_bl,)
        v_rot = uvw[:, 0] * sin_r + uvw[:, 1] * cos_r   # (n_bl,)

        for s_idx in range(n_src):
            l_s, m_s = lm[s_idx]
            # Phase: -2πi * (u*l + v*m) * f / c
            ul_vm = u_rot * l_s + v_rot * m_s              # (n_bl,)
            for f_idx in range(n_freq):
                phase = -2.0 * np.pi * ul_vm * freqs_hz[f_idx] / c
                v_model[:, f_idx, t_idx] += fluxes[s_idx, f_idx] * np.exp(1j * phase)

    return v_model


# ──────────────────────────────────────────────────────────────────────
#  Gain generation
# ──────────────────────────────────────────────────────────────────────

def generate_true_gains(n_ant: int, n_freq: int, n_time: int) -> np.ndarray:
    """
    Generate true complex gains per antenna, per frequency, per time.
    Amplitude ~ U(0.8, 1.2), phase ~ U(-30°, 30°).
    Gains vary smoothly across frequency and time via low-pass filtering.
    Shape: (n_ant, n_freq, n_time)
    """
    amplitudes = np.zeros((n_ant, n_freq, n_time))
    phases_deg = np.zeros((n_ant, n_freq, n_time))

    for a in range(n_ant):
        # Base amplitude and phase
        amp_base = RNG.uniform(0.8, 1.2)
        phase_base = RNG.uniform(-30.0, 30.0)

        # Small smooth variations across freq and time
        amp_var = RNG.normal(0, 0.03, size=(n_freq, n_time))
        phase_var = RNG.normal(0, 3.0, size=(n_freq, n_time))

        # Smooth via cumulative mean (simple low-pass)
        from scipy.ndimage import uniform_filter
        amp_var = uniform_filter(amp_var, size=3)
        phase_var = uniform_filter(phase_var, size=3)

        amplitudes[a] = amp_base + amp_var
        phases_deg[a] = phase_base + phase_var

    # Reference antenna: gain = 1+0j
    amplitudes[REF_ANT] = 1.0
    phases_deg[REF_ANT] = 0.0

    phases_rad = np.deg2rad(phases_deg)
    gains = amplitudes * np.exp(1j * phases_rad)
    return gains


# ──────────────────────────────────────────────────────────────────────
#  Forward model: corrupt visibilities
# ──────────────────────────────────────────────────────────────────────

def apply_gains(
    v_model: np.ndarray,
    gains: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Apply gains and add noise:
        V_obs_ij = G_i * V_model_ij * conj(G_j) + noise
    """
    n_bl, n_freq, n_time = v_model.shape

    v_obs = np.zeros_like(v_model)
    for bl_idx in range(n_bl):
        i, j = ant1[bl_idx], ant2[bl_idx]
        v_obs[bl_idx] = gains[i] * v_model[bl_idx] * np.conj(gains[j])

    # Add complex Gaussian noise
    signal_power = np.mean(np.abs(v_obs) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)  # per real/imag component
    noise = RNG.normal(0, noise_std, v_obs.shape) + 1j * RNG.normal(0, noise_std, v_obs.shape)
    v_obs += noise

    return v_obs


# ──────────────────────────────────────────────────────────────────────
#  Inverse solver: Stefcal algorithm
# ──────────────────────────────────────────────────────────────────────

def stefcal(
    v_obs: np.ndarray,
    v_model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    n_ant: int,
    max_iter: int = MAX_ITER,
    conv_tol: float = CONV_TOL,
    ref_ant: int = REF_ANT,
) -> tuple:
    """
    Stefcal (StefCal / Salvini & Wijnholds 2014) gain calibration.

    The measurement equation per baseline (p, q):
        V_obs_{pq} = g_p * V_model_{pq} * conj(g_q) + noise

    For antenna p, collecting all baselines where p appears:
        g_p^{new} = [ sum_q  V_obs_{pq} * V_model_{pq}^H * g_q^{old} ]
                   / [ sum_q  |V_model_{pq}|^2 * |g_q^{old}|^2 ]

    When baseline is stored as (i, j) with j=p, we use the conjugate:
        V_obs_{jp} = conj(V_obs_{pj}), V_model_{jp} = conj(V_model_{pj})

    Operates per frequency channel and time slot independently.
    """
    n_bl, n_freq, n_time = v_obs.shape

    # Initialize gains to unity
    g = np.ones((n_ant, n_freq, n_time), dtype=np.complex128)
    convergence = []

    for iteration in range(max_iter):
        g_old = g.copy()
        g_new = g.copy()

        for p in range(n_ant):
            if p == ref_ant:
                continue  # Fix reference antenna

            numerator = np.zeros((n_freq, n_time), dtype=np.complex128)
            denominator = np.zeros((n_freq, n_time), dtype=np.float64)

            for bl_idx in range(n_bl):
                i, j = ant1[bl_idx], ant2[bl_idx]

                if i == p:
                    # Baseline stored as (p, q) where q = j
                    # V_obs_{pq} = g_p * V_model_{pq} * conj(g_q)
                    # g_p = V_obs_{pq} * conj(g_q) * conj(V_model_{pq})
                    #       / (|g_q|^2 * |V_model_{pq}|^2)
                    q = j
                    z = v_model[bl_idx] * np.conj(g_old[q])  # V_model_{pq} * conj(g_q)
                    numerator += v_obs[bl_idx] * np.conj(z)
                    denominator += np.abs(z) ** 2

                elif j == p:
                    # Baseline stored as (i, q=p) — we need V_obs_{pq} where q=i
                    # Since V_obs_{ip} = g_i * V_model_{ip} * conj(g_p),
                    # the conjugate gives: conj(V_obs_{ip}) = conj(g_i) * conj(V_model_{ip}) * g_p
                    # i.e. V_obs_{pi} = conj(V_obs_{ip})
                    # So: g_p = V_obs_{pi} * conj(g_i) * conj(V_model_{pi})
                    #         / (|g_i|^2 * |V_model_{pi}|^2)
                    # With V_obs_{pi} = conj(V_obs_{ip}), V_model_{pi} = conj(V_model_{ip})
                    q = i
                    v_obs_pq = np.conj(v_obs[bl_idx])       # V_obs_{pi} = conj(V_obs_{ip})
                    v_mod_pq = np.conj(v_model[bl_idx])      # V_model_{pi} = conj(V_model_{ip})
                    z = v_mod_pq * np.conj(g_old[q])         # V_model_{pi} * conj(g_q=i)
                    numerator += v_obs_pq * np.conj(z)
                    denominator += np.abs(z) ** 2

            # Avoid division by zero
            safe_denom = np.where(denominator > 1e-30, denominator, 1e-30)
            g_new[p] = numerator / safe_denom

        # Damped update for stability
        damping = 0.5
        g = damping * g_new + (1.0 - damping) * g_old

        # Fix reference antenna
        g[ref_ant] = 1.0 + 0j

        # Check convergence
        rel_change = np.linalg.norm(g - g_old) / max(np.linalg.norm(g_old), 1e-30)
        convergence.append(rel_change)

        if rel_change < conv_tol:
            print(f"  Stefcal converged at iteration {iteration + 1}, "
                  f"rel_change = {rel_change:.2e}")
            break
    else:
        print(f"  Stefcal did not converge after {max_iter} iterations, "
              f"final rel_change = {convergence[-1]:.2e}")

    return g, convergence


# ──────────────────────────────────────────────────────────────────────
#  Resolve gain phase ambiguity
# ──────────────────────────────────────────────────────────────────────

def align_gains(g_cal: np.ndarray, g_true: np.ndarray, ref_ant: int = REF_ANT) -> np.ndarray:
    """
    Align calibrated gains to true gains by removing global phase offset.
    The reference antenna is already fixed, but there may be residual
    overall phase/amplitude ambiguity. We align using the mean phase offset
    across all antennas (excluding ref_ant).
    """
    # Global phase offset: ratio g_true / g_cal for each (ant, freq, time)
    # Then take the median phase of this ratio
    ratio = g_true / np.where(np.abs(g_cal) > 1e-15, g_cal, 1e-15)
    # Exclude reference antenna
    mask = np.ones(g_cal.shape[0], dtype=bool)
    mask[ref_ant] = False
    ratio_excl = ratio[mask]

    # Median phase offset
    phase_offset = np.angle(np.mean(ratio_excl))

    # Apply correction
    g_aligned = g_cal * np.exp(1j * phase_offset)
    return g_aligned


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(
    g_true: np.ndarray,
    g_cal: np.ndarray,
    v_obs: np.ndarray,
    v_model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
) -> dict:
    """Compute calibration quality metrics."""
    # Exclude reference antenna from gain metrics
    mask = np.ones(g_true.shape[0], dtype=bool)
    mask[REF_ANT] = False

    gt = g_true[mask]
    gc = g_cal[mask]

    # Gain amplitude RMSE
    amp_true = np.abs(gt).ravel()
    amp_cal = np.abs(gc).ravel()
    gain_amp_rmse = float(np.sqrt(np.mean((amp_true - amp_cal) ** 2)))

    # Gain phase RMSE (degrees)
    phase_true = np.angle(gt, deg=True).ravel()
    phase_cal = np.angle(gc, deg=True).ravel()
    # Handle phase wrapping
    phase_diff = phase_true - phase_cal
    phase_diff = (phase_diff + 180.0) % 360.0 - 180.0
    gain_phase_rmse = float(np.sqrt(np.mean(phase_diff ** 2)))

    # Gain correlation coefficient (complex)
    gt_flat = gt.ravel()
    gc_flat = gc.ravel()
    # CC on amplitudes
    cc_amp = float(np.corrcoef(amp_true, amp_cal)[0, 1])
    # CC on real/imag parts combined
    gt_ri = np.concatenate([gt_flat.real, gt_flat.imag])
    gc_ri = np.concatenate([gc_flat.real, gc_flat.imag])
    cc_complex = float(np.corrcoef(gt_ri, gc_ri)[0, 1])

    # Gain PSNR
    gain_mse = np.mean(np.abs(gt_flat - gc_flat) ** 2)
    gain_peak = np.max(np.abs(gt_flat)) ** 2
    gain_psnr = float(10.0 * np.log10(gain_peak / max(gain_mse, 1e-30)))

    # Corrected visibilities and residual PSNR
    n_bl = v_obs.shape[0]
    v_corrected = np.zeros_like(v_obs)
    for bl_idx in range(n_bl):
        i, j = ant1[bl_idx], ant2[bl_idx]
        v_corrected[bl_idx] = v_obs[bl_idx] / (g_cal[i] * np.conj(g_cal[j]))

    residual = v_corrected - v_model
    vis_mse = np.mean(np.abs(residual) ** 2)
    vis_peak = np.max(np.abs(v_model)) ** 2
    vis_psnr = float(10.0 * np.log10(vis_peak / max(vis_mse, 1e-30)))

    return {
        "gain_amp_RMSE": round(gain_amp_rmse, 6),
        "gain_phase_RMSE_deg": round(gain_phase_rmse, 4),
        "gain_CC_amplitude": round(cc_amp, 6),
        "gain_CC_complex": round(cc_complex, 6),
        "gain_PSNR_dB": round(gain_psnr, 2),
        "visibility_residual_PSNR_dB": round(vis_psnr, 2),
        "PSNR": round(vis_psnr, 2),
    }


# ──────────────────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────────────────

def create_visualization(
    g_true: np.ndarray,
    g_cal: np.ndarray,
    v_obs: np.ndarray,
    v_model: np.ndarray,
    v_corrected: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    convergence: list,
    save_path: str,
) -> None:
    """Generate a 2×2 visualization figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel 1: Gain amplitudes ──
    ax = axes[0, 0]
    # Average over freq and time for each antenna
    amp_true_ant = np.abs(g_true).mean(axis=(1, 2))
    amp_cal_ant = np.abs(g_cal).mean(axis=(1, 2))
    ant_ids = np.arange(g_true.shape[0])
    width = 0.35
    ax.bar(ant_ids - width / 2, amp_true_ant, width, label='True', color='steelblue', alpha=0.8)
    ax.bar(ant_ids + width / 2, amp_cal_ant, width, label='Calibrated', color='coral', alpha=0.8)
    ax.set_xlabel('Antenna ID')
    ax.set_ylabel('Gain Amplitude')
    ax.set_title('Gain Amplitudes: True vs Calibrated')
    ax.legend()
    ax.set_xticks(ant_ids)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: Gain phases ──
    ax = axes[0, 1]
    phase_true_ant = np.angle(g_true, deg=True).mean(axis=(1, 2))
    phase_cal_ant = np.angle(g_cal, deg=True).mean(axis=(1, 2))
    ax.bar(ant_ids - width / 2, phase_true_ant, width, label='True', color='steelblue', alpha=0.8)
    ax.bar(ant_ids + width / 2, phase_cal_ant, width, label='Calibrated', color='coral', alpha=0.8)
    ax.set_xlabel('Antenna ID')
    ax.set_ylabel('Gain Phase (degrees)')
    ax.set_title('Gain Phases: True vs Calibrated')
    ax.legend()
    ax.set_xticks(ant_ids)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 3: Visibility amplitudes (before / after calibration) ──
    ax = axes[1, 0]
    # Pick a representative baseline and frequency
    bl_show = 0
    freq_show = N_FREQ // 2
    time_axis = np.arange(N_TIME)

    v_obs_line = np.abs(v_obs[bl_show, freq_show, :])
    v_model_line = np.abs(v_model[bl_show, freq_show, :])
    v_corr_line = np.abs(v_corrected[bl_show, freq_show, :])

    ax.plot(time_axis, v_model_line, 'k-', lw=2, label='Model', alpha=0.7)
    ax.plot(time_axis, v_obs_line, 'r--', lw=1.2, label='Observed (corrupted)', alpha=0.7)
    ax.plot(time_axis, v_corr_line, 'g-', lw=1.5, label='Corrected', alpha=0.9)
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Visibility Amplitude')
    ax.set_title(f'Visibility Amp (baseline {ant1[bl_show]}-{ant2[bl_show]}, freq ch {freq_show})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 4: Convergence ──
    ax = axes[1, 1]
    ax.semilogy(convergence, 'b-', lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Change')
    ax.set_title('Stefcal Convergence')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=CONV_TOL, color='r', linestyle='--', alpha=0.5, label=f'Tolerance = {CONV_TOL:.0e}')
    ax.legend()

    plt.suptitle(
        'Radio Interferometry Gain Calibration (Stefcal)\n'
        f'{N_ANT} antennas, {N_FREQ} freq channels, {N_TIME} time slots, SNR={SNR_DB} dB',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Visualization saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Radio Interferometry Gain Calibration (Stefcal)")
    print("=" * 60)

    # 1. Generate sky model
    print("\n[1/6] Generating sky model ...")
    fluxes, lm, freqs = generate_sky_model(N_SRC, N_FREQ)
    print(f"  {N_SRC} point sources, {N_FREQ} frequency channels "
          f"({freqs[0]:.2f} - {freqs[-1]:.2f} GHz)")

    # 2. Generate antenna layout and baselines
    print("[2/6] Setting up array geometry ...")
    positions = generate_antenna_layout(N_ANT)
    ant1, ant2, uvw = compute_baselines(positions)
    n_bl = len(ant1)
    print(f"  {N_ANT} antennas, {n_bl} baselines")

    # 3. Compute model visibilities
    print("[3/6] Computing model visibilities ...")
    v_model = compute_model_visibilities(fluxes, lm, uvw, freqs, N_TIME)
    print(f"  V_model shape: {v_model.shape}")

    # 4. Generate true gains and corrupt visibilities
    print("[4/6] Generating true gains & corrupting visibilities ...")
    g_true = generate_true_gains(N_ANT, N_FREQ, N_TIME)
    v_obs = apply_gains(v_model, g_true, ant1, ant2, SNR_DB)
    print(f"  Gains shape: {g_true.shape}, SNR = {SNR_DB} dB")

    # 5. Run Stefcal calibration
    print("[5/6] Running Stefcal calibration ...")
    g_cal, convergence = stefcal(v_obs, v_model, ant1, ant2, N_ANT)

    # Align gains (resolve phase ambiguity)
    g_cal = align_gains(g_cal, g_true, REF_ANT)
    print(f"  Calibration completed in {len(convergence)} iterations")

    # 6. Compute metrics
    print("[6/6] Computing metrics ...")
    metrics = compute_metrics(g_true, g_cal, v_obs, v_model, ant1, ant2)

    print("\n  ── Calibration Results ──")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    # Corrected visibilities for visualization
    n_bl_vis = v_obs.shape[0]
    v_corrected = np.zeros_like(v_obs)
    for bl_idx in range(n_bl_vis):
        i, j = ant1[bl_idx], ant2[bl_idx]
        v_corrected[bl_idx] = v_obs[bl_idx] / (g_cal[i] * np.conj(g_cal[j]))

    # Save outputs
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    gt_path = os.path.join(RESULTS_DIR, "ground_truth.npy")
    recon_path = os.path.join(RESULTS_DIR, "reconstruction.npy")
    np.save(gt_path, g_true)
    np.save(recon_path, g_cal)
    print(f"  Ground truth saved to {gt_path}")
    print(f"  Reconstruction saved to {recon_path}")

    # Visualization
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    create_visualization(
        g_true, g_cal, v_obs, v_model, v_corrected,
        ant1, ant2, convergence, vis_path,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
