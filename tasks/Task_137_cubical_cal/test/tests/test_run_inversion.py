import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the agent's function
from agent_run_inversion import run_inversion

# ── Inject the Referee (evaluate_results) verbatim from Reference B ──

CONV_TOL = 1e-7

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(
    g_true: np.ndarray,
    g_cal: np.ndarray,
    v_obs: np.ndarray,
    v_model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    convergence: list,
    ref_ant: int,
    results_dir: str,
    n_ant: int,
    n_freq: int,
    n_time: int,
    snr_db: float,
) -> dict:
    """
    Evaluate calibration quality and create visualizations.
    """
    # Compute corrected visibilities
    n_bl = v_obs.shape[0]
    v_corrected = np.zeros_like(v_obs)
    for bl_idx in range(n_bl):
        i, j = ant1[bl_idx], ant2[bl_idx]
        denom = g_cal[i] * np.conj(g_cal[j])
        denom = np.where(np.abs(denom) > 1e-15, denom, 1e-15)
        v_corrected[bl_idx] = v_obs[bl_idx] / denom

    # Compute metrics
    mask = np.ones(g_true.shape[0], dtype=bool)
    mask[ref_ant] = False

    gt = g_true[mask]
    gc = g_cal[mask]

    # Gain amplitude RMSE
    amp_true = np.abs(gt).ravel()
    amp_cal = np.abs(gc).ravel()
    gain_amp_rmse = float(np.sqrt(np.mean((amp_true - amp_cal) ** 2)))

    # Gain phase RMSE (degrees)
    phase_true = np.angle(gt, deg=True).ravel()
    phase_cal = np.angle(gc, deg=True).ravel()
    phase_diff = phase_true - phase_cal
    phase_diff = (phase_diff + 180.0) % 360.0 - 180.0
    gain_phase_rmse = float(np.sqrt(np.mean(phase_diff ** 2)))

    # Correlation coefficients
    gt_flat = gt.ravel()
    gc_flat = gc.ravel()
    cc_amp = float(np.corrcoef(amp_true, amp_cal)[0, 1])
    gt_ri = np.concatenate([gt_flat.real, gt_flat.imag])
    gc_ri = np.concatenate([gc_flat.real, gc_flat.imag])
    cc_complex = float(np.corrcoef(gt_ri, gc_ri)[0, 1])

    # Gain PSNR
    gain_mse = np.mean(np.abs(gt_flat - gc_flat) ** 2)
    gain_peak = np.max(np.abs(gt_flat)) ** 2
    gain_psnr = float(10.0 * np.log10(gain_peak / max(gain_mse, 1e-30)))

    # Visibility residual PSNR
    residual = v_corrected - v_model
    vis_mse = np.mean(np.abs(residual) ** 2)
    vis_peak = np.max(np.abs(v_model)) ** 2
    vis_psnr = float(10.0 * np.log10(vis_peak / max(vis_mse, 1e-30)))

    metrics = {
        "gain_amp_RMSE": round(gain_amp_rmse, 6),
        "gain_phase_RMSE_deg": round(gain_phase_rmse, 4),
        "gain_CC_amplitude": round(cc_amp, 6),
        "gain_CC_complex": round(cc_complex, 6),
        "gain_PSNR_dB": round(gain_psnr, 2),
        "visibility_residual_PSNR_dB": round(vis_psnr, 2),
        "PSNR": round(vis_psnr, 2),
    }

    print("\n  ── Calibration Results ──")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # Save arrays
    gt_path = os.path.join(results_dir, "ground_truth.npy")
    recon_path = os.path.join(results_dir, "reconstruction.npy")
    np.save(gt_path, g_true)
    np.save(recon_path, g_cal)
    print(f"  Ground truth saved to {gt_path}")
    print(f"  Reconstruction saved to {recon_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Gain amplitudes
    ax = axes[0, 0]
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

    # Panel 2: Gain phases
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

    # Panel 3: Visibility amplitudes
    ax = axes[1, 0]
    bl_show = 0
    freq_show = n_freq // 2
    time_axis = np.arange(n_time)

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

    # Panel 4: Convergence
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
        f'{n_ant} antennas, {n_freq} freq channels, {n_time} time slots, SNR={snr_db} dB',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Visualization saved to {vis_path}")

    return metrics


# ── Main Test Logic ──

def main():
    data_paths = ['/data/yjh/cubical_cal_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer vs inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Check for chained (inner) execution
    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("\nPattern 2: Chained Execution detected.")
        print("Running outer function (run_inversion) to get operator...")
        agent_operator = run_inversion(*outer_args, **outer_kwargs)

        for ip in inner_paths:
            print(f"\nLoading inner data from: {ip}")
            with open(ip, 'rb') as f:
                inner_data = dill.load(f)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)

            print("Running operator with inner data...")
            agent_result = agent_operator(*inner_args, **inner_kwargs)

            # For chained, evaluate the inner result
            # This path is unlikely for this problem, but handled for completeness
            print("Chained execution complete.")
    else:
        # Pattern 1: Direct Execution
        print("\nPattern 1: Direct Execution.")
        print("Running agent's run_inversion...")
        agent_result = run_inversion(*outer_args, **outer_kwargs)
        std_result = std_output

    # ── Extract necessary parameters for evaluate_results ──
    # We need: g_true, v_obs, v_model, ant1, ant2, ref_ant, n_ant, n_freq, n_time, snr_db
    # These come from the input arguments

    # Parse inputs - run_inversion signature:
    # (v_obs, v_model, ant1, ant2, n_ant, g_true, max_iter, conv_tol, ref_ant)
    if len(outer_args) >= 9:
        v_obs = outer_args[0]
        v_model = outer_args[1]
        ant1 = outer_args[2]
        ant2 = outer_args[3]
        n_ant = outer_args[4]
        g_true = outer_args[5]
        max_iter = outer_args[6]
        conv_tol = outer_args[7]
        ref_ant = outer_args[8]
    else:
        # Try kwargs
        v_obs = outer_kwargs.get('v_obs', outer_args[0] if len(outer_args) > 0 else None)
        v_model = outer_kwargs.get('v_model', outer_args[1] if len(outer_args) > 1 else None)
        ant1 = outer_kwargs.get('ant1', outer_args[2] if len(outer_args) > 2 else None)
        ant2 = outer_kwargs.get('ant2', outer_args[3] if len(outer_args) > 3 else None)
        n_ant = outer_kwargs.get('n_ant', outer_args[4] if len(outer_args) > 4 else None)
        g_true = outer_kwargs.get('g_true', outer_args[5] if len(outer_args) > 5 else None)
        max_iter = outer_kwargs.get('max_iter', outer_args[6] if len(outer_args) > 6 else None)
        conv_tol = outer_kwargs.get('conv_tol', outer_args[7] if len(outer_args) > 7 else None)
        ref_ant = outer_kwargs.get('ref_ant', outer_args[8] if len(outer_args) > 8 else None)

    n_bl, n_freq, n_time = v_obs.shape

    # Estimate SNR (not stored, use a default)
    snr_db = 30.0  # Default; exact value only affects plot title

    # ── Evaluate Agent Result ──
    print("\n" + "=" * 60)
    print("Evaluating AGENT result...")
    print("=" * 60)

    agent_results_dir = os.path.join(RESULTS_DIR, "agent")
    os.makedirs(agent_results_dir, exist_ok=True)

    agent_g_cal = agent_result['g_cal']
    agent_convergence = agent_result['convergence']

    agent_metrics = evaluate_results(
        g_true=g_true,
        g_cal=agent_g_cal,
        v_obs=v_obs,
        v_model=v_model,
        ant1=ant1,
        ant2=ant2,
        convergence=agent_convergence,
        ref_ant=ref_ant,
        results_dir=agent_results_dir,
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        snr_db=snr_db,
    )

    # ── Evaluate Standard Result ──
    print("\n" + "=" * 60)
    print("Evaluating STANDARD (ground truth) result...")
    print("=" * 60)

    std_results_dir = os.path.join(RESULTS_DIR, "standard")
    os.makedirs(std_results_dir, exist_ok=True)

    std_g_cal = std_result['g_cal']
    std_convergence = std_result['convergence']

    std_metrics = evaluate_results(
        g_true=g_true,
        g_cal=std_g_cal,
        v_obs=v_obs,
        v_model=v_model,
        ant1=ant1,
        ant2=ant2,
        convergence=std_convergence,
        ref_ant=ref_ant,
        results_dir=std_results_dir,
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        snr_db=snr_db,
    )

    # ── Compare Scores ──
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Primary metric: PSNR (visibility_residual_PSNR_dB) - Higher is better
    agent_psnr = agent_metrics.get("PSNR", agent_metrics.get("visibility_residual_PSNR_dB", 0.0))
    std_psnr = std_metrics.get("PSNR", std_metrics.get("visibility_residual_PSNR_dB", 0.0))

    agent_gain_psnr = agent_metrics.get("gain_PSNR_dB", 0.0)
    std_gain_psnr = std_metrics.get("gain_PSNR_dB", 0.0)

    agent_amp_rmse = agent_metrics.get("gain_amp_RMSE", float('inf'))
    std_amp_rmse = std_metrics.get("gain_amp_RMSE", float('inf'))

    agent_phase_rmse = agent_metrics.get("gain_phase_RMSE_deg", float('inf'))
    std_phase_rmse = std_metrics.get("gain_phase_RMSE_deg", float('inf'))

    agent_cc_complex = agent_metrics.get("gain_CC_complex", 0.0)
    std_cc_complex = std_metrics.get("gain_CC_complex", 0.0)

    print(f"\n  Visibility PSNR  -> Agent: {agent_psnr:.2f} dB, Standard: {std_psnr:.2f} dB")
    print(f"  Gain PSNR        -> Agent: {agent_gain_psnr:.2f} dB, Standard: {std_gain_psnr:.2f} dB")
    print(f"  Gain Amp RMSE    -> Agent: {agent_amp_rmse:.6f}, Standard: {std_amp_rmse:.6f}")
    print(f"  Gain Phase RMSE  -> Agent: {agent_phase_rmse:.4f} deg, Standard: {std_phase_rmse:.4f} deg")
    print(f"  Gain CC Complex  -> Agent: {agent_cc_complex:.6f}, Standard: {std_cc_complex:.6f}")

    # ── Determine Pass/Fail ──
    # PSNR is "higher is better". Allow 10% margin (relative to std).
    # Also check that key metrics are not drastically worse.

    passed = True
    reasons = []

    # Check visibility PSNR (primary metric)
    if std_psnr > 0:
        psnr_ratio = agent_psnr / std_psnr
        if psnr_ratio < 0.90:
            passed = False
            reasons.append(f"Visibility PSNR too low: {agent_psnr:.2f} vs {std_psnr:.2f} (ratio={psnr_ratio:.3f})")
    else:
        # If std_psnr is 0 or negative, just check agent is not much worse
        if agent_psnr < std_psnr - 3.0:
            passed = False
            reasons.append(f"Visibility PSNR degraded: {agent_psnr:.2f} vs {std_psnr:.2f}")

    # Check gain PSNR
    if std_gain_psnr > 0:
        gain_psnr_ratio = agent_gain_psnr / std_gain_psnr
        if gain_psnr_ratio < 0.90:
            passed = False
            reasons.append(f"Gain PSNR too low: {agent_gain_psnr:.2f} vs {std_gain_psnr:.2f} (ratio={gain_psnr_ratio:.3f})")

    # Check amplitude RMSE (lower is better) - allow 10% margin
    if std_amp_rmse > 0:
        if agent_amp_rmse > std_amp_rmse * 1.10:
            passed = False
            reasons.append(f"Gain amp RMSE too high: {agent_amp_rmse:.6f} vs {std_amp_rmse:.6f}")

    # Check complex correlation (higher is better)
    if std_cc_complex > 0:
        if agent_cc_complex < std_cc_complex * 0.90:
            passed = False
            reasons.append(f"Complex correlation too low: {agent_cc_complex:.6f} vs {std_cc_complex:.6f}")

    print("\n" + "=" * 60)
    if passed:
        print("TEST PASSED: Agent performance is within acceptable bounds.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("TEST FAILED: Agent performance degraded significantly.")
        for r in reasons:
            print(f"  - {r}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR during test execution:\n{traceback.format_exc()}")
        sys.exit(1)