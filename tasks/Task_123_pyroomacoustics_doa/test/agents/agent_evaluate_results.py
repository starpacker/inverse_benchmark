import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

def evaluate_results(
    inversion_result: dict,
    data: dict,
    nfft: int
) -> dict:
    """
    Evaluate DOA estimation results by computing metrics and creating visualizations.
    
    Args:
        inversion_result: Dictionary from run_inversion containing estimated DOAs
        data: Dictionary from load_and_preprocess_data containing ground truth
        nfft: FFT size used
    
    Returns:
        Dictionary containing all computed metrics
    """
    print("\nStep 6: Computing metrics...")
    
    # Extract values
    true_azimuths_deg = data["true_azimuths_deg"]
    estimated_azimuths_deg = inversion_result["estimated_azimuths_deg"]
    spatial_spectrum = inversion_result["spatial_spectrum"]
    azimuth_grid_deg = inversion_result["azimuth_grid_deg"]
    azimuth_grid = inversion_result["azimuth_grid"]
    peak_indices = inversion_result["peak_indices"]
    srp_spectrum = inversion_result["srp_spectrum"]
    estimated_azimuths_rad = inversion_result["estimated_azimuths_rad"]
    true_azimuths_rad = data["true_azimuths_rad"]
    
    # Helper function for angular distance
    def angular_distance(a1, a2):
        """Compute minimum angular distance accounting for wraparound."""
        diff = abs(a1 - a2) % 360
        return min(diff, 360 - diff)
    
    # Greedy matching: for each true azimuth, find closest estimated
    est_list = list(estimated_azimuths_deg.copy())
    matched_errors = []
    for true_az in true_azimuths_deg:
        distances = [angular_distance(true_az, est) for est in est_list]
        best_idx = np.argmin(distances)
        matched_errors.append(distances[best_idx])
        est_list.pop(best_idx)
    
    angular_errors = np.array(matched_errors)
    mean_angular_error = np.mean(angular_errors)
    max_angular_error = np.max(angular_errors)
    rmse_angular = np.sqrt(np.mean(angular_errors ** 2))
    
    print(f"  True azimuths (deg):      {[f'{a:.1f}' for a in true_azimuths_deg]}")
    print(f"  Angular errors per source (deg): {[f'{e:.2f}' for e in angular_errors]}")
    print(f"  Mean angular error: {mean_angular_error:.2f}°")
    print(f"  Max angular error: {max_angular_error:.2f}°")
    print(f"  RMSE angular: {rmse_angular:.2f}°")
    
    # Construct ground-truth angular profile (sum of Gaussians at true azimuths)
    sigma_deg = 5.0
    gt_profile = np.zeros_like(azimuth_grid_deg)
    for true_az in true_azimuths_deg:
        diff = np.abs(azimuth_grid_deg - true_az)
        diff = np.minimum(diff, 360 - diff)
        gt_profile += np.exp(-0.5 * (diff / sigma_deg) ** 2)
    
    # Gaussian-smooth the MUSIC spectrum
    spatial_spectrum_smoothed = gaussian_filter1d(spatial_spectrum, sigma=4, mode='wrap')
    
    # Normalize both spectra to [0, 1]
    gt_norm = gt_profile / (np.max(gt_profile) + 1e-12)
    recon_norm = spatial_spectrum_smoothed / (np.max(spatial_spectrum_smoothed) + 1e-12)
    
    # PSNR of spatial spectrum
    mse = np.mean((gt_norm - recon_norm) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))
    print(f"  Spatial spectrum PSNR: {psnr:.2f} dB")
    
    # Correlation coefficient
    cc = np.corrcoef(gt_norm, recon_norm)[0, 1]
    print(f"  Correlation coefficient: {cc:.4f}")
    
    # Step 7: Save results
    print("\nStep 7: Saving results...")
    os.makedirs("results", exist_ok=True)
    
    metrics = {
        "task": "pyroomacoustics_doa",
        "task_number": 123,
        "inverse_problem": "Direction-of-Arrival estimation from microphone array recordings",
        "algorithm": "MUSIC (MUltiple SIgnal Classification)",
        "n_sources": data["n_sources"],
        "n_microphones": data["n_mics"],
        "true_azimuths_deg": [round(a, 2) for a in true_azimuths_deg.tolist()],
        "estimated_azimuths_deg": [round(a, 2) for a in estimated_azimuths_deg.tolist()],
        "angular_errors_deg": [round(e, 2) for e in angular_errors.tolist()],
        "mean_angular_error_deg": round(float(mean_angular_error), 2),
        "max_angular_error_deg": round(float(max_angular_error), 2),
        "rmse_angular_deg": round(float(rmse_angular), 2),
        "spatial_spectrum_psnr_db": round(float(psnr), 2),
        "correlation_coefficient": round(float(cc), 4),
        "snr_db": data["snr_db"],
        "fs_hz": data["fs"],
        "nfft": nfft,
        "room_dim_m": data["room_dim"],
        "mic_radius_m": data["mic_radius"],
    }
    
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: results/metrics.json")
    
    # Save reconstruction and ground truth
    np.save("results/reconstruction.npy", recon_norm)
    np.save("results/ground_truth.npy", gt_norm)
    np.save("gt_output.npy", gt_norm)
    np.save("recon_output.npy", recon_norm)
    print("  Saved: results/reconstruction.npy (normalized spatial spectrum)")
    print("  Saved: results/ground_truth.npy (normalized ground-truth profile)")
    print("  Saved: gt_output.npy, recon_output.npy")
    
    # Step 8: Visualization
    print("\nStep 8: Creating visualization...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # Panel 1: MUSIC Spatial Spectrum
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(azimuth_grid_deg, recon_norm, 'b-', linewidth=1.5, label='MUSIC spectrum')
    ax1.plot(azimuth_grid_deg, gt_norm, 'r--', linewidth=1.5, alpha=0.7, label='Ground truth profile')
    for i, az in enumerate(true_azimuths_deg):
        ax1.axvline(az, color='red', linestyle=':', alpha=0.6,
                    label=f'True src {i+1}: {az:.1f}°')
    for i, az in enumerate(estimated_azimuths_deg):
        ax1.axvline(az, color='green', linestyle='--', alpha=0.6,
                    label=f'Est src {i+1}: {az:.1f}°')
    ax1.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax1.set_ylabel('Normalized Power', fontsize=12)
    ax1.set_title('MUSIC Spatial Spectrum (DOA Estimation)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim([0, 360])
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Polar plot of DOA
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    theta = np.radians(azimuth_grid_deg)
    ax2.plot(theta, recon_norm, 'b-', linewidth=1.5, label='MUSIC')
    ax2.plot(theta, srp_spectrum / (np.max(srp_spectrum) + 1e-12), 'g-',
             linewidth=1.0, alpha=0.5, label='SRP-PHAT')
    for i, az in enumerate(true_azimuths_rad):
        ax2.plot(az, 1.0, 'r^', markersize=12, label=f'True {i+1}')
    for i, az_rad in enumerate(estimated_azimuths_rad):
        ax2.plot(az_rad, recon_norm[peak_indices[i]], 'gs', markersize=10,
                 label=f'Est {i+1}')
    ax2.set_title('Polar DOA Spectrum', fontsize=13, fontweight='bold', pad=20)
    ax2.legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.3, 0))
    
    # Panel 3: Room geometry and source/mic positions
    ax3 = fig.add_subplot(2, 2, 3)
    room_dim = data["room_dim"]
    mic_locs = data["mic_locs"]
    array_center = data["array_center"]
    source_positions = data["source_positions"]
    
    room_rect = plt.Rectangle((0, 0), room_dim[0], room_dim[1],
                               fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(room_rect)
    ax3.scatter(mic_locs[0, :], mic_locs[1, :], c='blue', marker='o', s=60,
                zorder=5, label='Microphones')
    ax3.scatter(*array_center, c='blue', marker='+', s=200, linewidths=2, zorder=5)
    for i, pos in enumerate(source_positions):
        ax3.scatter(*pos, c='red', marker='*', s=200, zorder=5,
                    label=f'Source {i+1} ({true_azimuths_deg[i]:.1f}°)')
        ax3.annotate(f'S{i+1}', xy=pos, xytext=(pos[0]+0.15, pos[1]+0.15),
                     fontsize=10, color='red', fontweight='bold')
    for pos in source_positions:
        ax3.plot([array_center[0], pos[0]], [array_center[1], pos[1]],
                 'r--', alpha=0.3)
    ax3.set_xlim([-0.5, room_dim[0] + 0.5])
    ax3.set_ylim([-0.5, room_dim[1] + 0.5])
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Y (m)', fontsize=12)
    ax3.set_title('Room Geometry & Source Positions', fontsize=13, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Time-domain signals
    ax4 = fig.add_subplot(2, 2, 4)
    noisy_signals = data["noisy_signals"]
    fs = data["fs"]
    t = np.arange(noisy_signals.shape[1]) / fs * 1000  # ms
    n_plot = min(3, data["n_mics"])
    for i in range(n_plot):
        offset = i * 0.3
        ax4.plot(t[:800], noisy_signals[i, :800] / np.max(np.abs(noisy_signals)) + offset,
                 linewidth=0.7, label=f'Mic {i+1}')
    ax4.set_xlabel('Time (ms)', fontsize=12)
    ax4.set_ylabel('Normalized Amplitude (offset)', fontsize=12)
    ax4.set_title('Multi-Channel Recordings (first 50ms)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(
        f'Task 123: DOA Estimation via MUSIC | '
        f'Mean Angular Error: {mean_angular_error:.2f}° | '
        f'Spectrum PSNR: {psnr:.2f} dB | CC: {cc:.4f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/reconstruction_result.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/reconstruction_result.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  True azimuths:      {[f'{a:.1f}°' for a in true_azimuths_deg]}")
    print(f"  Estimated azimuths: {[f'{a:.1f}°' for a in estimated_azimuths_deg]}")
    print(f"  Angular errors:     {[f'{e:.2f}°' for e in angular_errors]}")
    print(f"  Mean angular error: {mean_angular_error:.2f}°")
    print(f"  RMSE angular:       {rmse_angular:.2f}°")
    print(f"  Spatial spectrum PSNR: {psnr:.2f} dB")
    print(f"  Correlation coefficient: {cc:.4f}")
    print("=" * 60)
    
    return metrics
