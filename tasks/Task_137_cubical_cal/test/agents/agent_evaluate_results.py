import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import json

import os

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
    
    Computes metrics:
    - Gain amplitude RMSE
    - Gain phase RMSE
    - Correlation coefficients
    - PSNR values
    
    Creates visualization and saves results.
    
    Returns:
        Dictionary of metrics
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
