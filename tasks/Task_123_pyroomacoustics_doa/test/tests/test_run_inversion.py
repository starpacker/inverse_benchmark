import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

# Import the target function
from agent_run_inversion import run_inversion


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


def main():
    data_paths = ['/data/yjh/pyroomacoustics_doa_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data paths
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
    
    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function name: {func_name}")
    print(f"Number of positional args: {len(args)}")
    print(f"Keyword args keys: {list(kwargs.keys())}")
    
    # Determine if chained execution
    is_chained = len(inner_paths) > 0
    
    if is_chained:
        print("\n--- Chained Execution Mode ---")
        # Run outer to get operator
        print("Running run_inversion (outer) to get operator...")
        agent_operator = run_inversion(*args, **kwargs)
        
        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        inner_std_output = inner_data.get('output', None)
        
        # Run operator with inner args
        print("Running operator (inner)...")
        agent_result = agent_operator(*inner_args, **inner_kwargs)
        std_result = inner_std_output
    else:
        print("\n--- Direct Execution Mode ---")
        # Run agent
        print("Running run_inversion...")
        np.random.seed(42)
        agent_result = run_inversion(*args, **kwargs)
        std_result = std_output
    
    # Now we need the 'data' dict for evaluate_results
    # The evaluate_results function needs a 'data' dict with ground truth info.
    # We need to find/construct this from the broader pipeline data.
    # Let's check if there's a load_and_preprocess_data pkl file
    std_data_dir = os.path.dirname(outer_path)
    preprocess_data_path = None
    for fname in os.listdir(std_data_dir):
        if 'load_and_preprocess' in fname and fname.endswith('.pkl'):
            preprocess_data_path = os.path.join(std_data_dir, fname)
            break
    
    # Also check for any other data files that might contain ground truth
    all_pkls = []
    for fname in os.listdir(std_data_dir):
        if fname.endswith('.pkl'):
            all_pkls.append(fname)
    print(f"Available pkl files in std_data dir: {all_pkls}")
    
    data_for_eval = None
    
    if preprocess_data_path and os.path.exists(preprocess_data_path):
        print(f"Loading preprocess data from: {preprocess_data_path}")
        with open(preprocess_data_path, 'rb') as f:
            preprocess_pkl = dill.load(f)
        data_for_eval = preprocess_pkl.get('output', None)
        if data_for_eval is None:
            data_for_eval = preprocess_pkl
    
    # Extract nfft from the kwargs or args
    # run_inversion signature: (X, mic_locs, fs, nfft, c, n_sources, freq_range)
    import inspect
    sig = inspect.signature(run_inversion)
    param_names = list(sig.parameters.keys())
    
    # Build a complete argument dict
    bound_args = {}
    for i, val in enumerate(args):
        if i < len(param_names):
            bound_args[param_names[i]] = val
    bound_args.update(kwargs)
    
    nfft = bound_args.get('nfft', 1024)
    fs = bound_args.get('fs', 16000)
    mic_locs = bound_args.get('mic_locs', None)
    n_sources = bound_args.get('n_sources', 2)
    
    print(f"nfft: {nfft}, fs: {fs}, n_sources: {n_sources}")
    
    # If we don't have preprocess data, we need to construct a minimal 'data' dict
    # for evaluate_results from whatever info we have
    if data_for_eval is None:
        print("WARNING: No preprocess data found. Attempting to construct minimal data dict.")
        # We'll try to find any pkl that might have the ground truth
        for fname in all_pkls:
            if fname != os.path.basename(outer_path):
                try:
                    fpath = os.path.join(std_data_dir, fname)
                    with open(fpath, 'rb') as f:
                        candidate = dill.load(f)
                    if isinstance(candidate, dict):
                        out = candidate.get('output', candidate)
                        if isinstance(out, dict) and 'true_azimuths_deg' in out:
                            data_for_eval = out
                            print(f"Found ground truth data in: {fname}")
                            break
                except Exception as e:
                    print(f"  Could not load {fname}: {e}")
    
    if data_for_eval is None:
        # Last resort: compare the agent result vs std result directly using key metrics
        print("WARNING: Cannot find ground truth data for full evaluate_results.")
        print("Falling back to direct comparison of agent vs standard output.")
        
        # Compare key output fields
        try:
            # Compare estimated azimuths
            agent_az = np.sort(agent_result['estimated_azimuths_deg'])
            std_az = np.sort(std_result['estimated_azimuths_deg'])
            
            print(f"\nAgent estimated azimuths (sorted): {agent_az}")
            print(f"Standard estimated azimuths (sorted): {std_az}")
            
            # Check angular difference
            az_diffs = []
            for a, s in zip(agent_az, std_az):
                diff = abs(a - s) % 360
                diff = min(diff, 360 - diff)
                az_diffs.append(diff)
            
            max_az_diff = max(az_diffs) if az_diffs else 0
            mean_az_diff = np.mean(az_diffs) if az_diffs else 0
            
            print(f"Max azimuth difference: {max_az_diff:.2f}°")
            print(f"Mean azimuth difference: {mean_az_diff:.2f}°")
            
            # Compare spatial spectra correlation
            agent_spec = agent_result['spatial_spectrum']
            std_spec = std_result['spatial_spectrum']
            
            # Normalize
            agent_spec_norm = agent_spec / (np.max(agent_spec) + 1e-12)
            std_spec_norm = std_spec / (np.max(std_spec) + 1e-12)
            
            cc = np.corrcoef(agent_spec_norm, std_spec_norm)[0, 1]
            print(f"Spatial spectrum correlation: {cc:.6f}")
            
            # SRP correlation
            agent_srp = agent_result['srp_spectrum']
            std_srp = std_result['srp_spectrum']
            agent_srp_norm = agent_srp / (np.max(agent_srp) + 1e-12)
            std_srp_norm = std_srp / (np.max(std_srp) + 1e-12)
            srp_cc = np.corrcoef(agent_srp_norm, std_srp_norm)[0, 1]
            print(f"SRP spectrum correlation: {srp_cc:.6f}")
            
            # Acceptance criteria
            # Azimuths should be very close (within 5 degrees)
            # Spectra should be highly correlated (> 0.9)
            success = True
            if max_az_diff > 10.0:
                print(f"FAIL: Max azimuth difference {max_az_diff:.2f}° > 10.0°")
                success = False
            if cc < 0.9:
                print(f"FAIL: Spatial spectrum correlation {cc:.4f} < 0.9")
                success = False
            
            if success:
                print("\nPASS: Agent output matches standard output within tolerance.")
                sys.exit(0)
            else:
                print("\nFAIL: Agent output deviates significantly from standard.")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR during direct comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # We have data_for_eval, run full evaluation
    print("\n--- Running evaluate_results for AGENT output ---")
    try:
        metrics_agent = evaluate_results(agent_result, data_for_eval, nfft)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n--- Running evaluate_results for STANDARD output ---")
    try:
        metrics_std = evaluate_results(std_result, data_for_eval, nfft)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract key metrics for comparison
    agent_psnr = metrics_agent.get('spatial_spectrum_psnr_db', 0)
    std_psnr = metrics_std.get('spatial_spectrum_psnr_db', 0)
    agent_cc = metrics_agent.get('correlation_coefficient', 0)
    std_cc = metrics_std.get('correlation_coefficient', 0)
    agent_mae = metrics_agent.get('mean_angular_error_deg', 999)
    std_mae = metrics_std.get('mean_angular_error_deg', 999)
    agent_rmse = metrics_agent.get('rmse_angular_deg', 999)
    std_rmse = metrics_std.get('rmse_angular_deg', 999)
    
    print("\n" + "=" * 60)
    print("COMPARISON: Agent vs Standard")
    print("=" * 60)
    print(f"  PSNR          -> Agent: {agent_psnr:.2f} dB,  Standard: {std_psnr:.2f} dB")
    print(f"  Correlation   -> Agent: {agent_cc:.4f},   Standard: {std_cc:.4f}")
    print(f"  Mean Ang Err  -> Agent: {agent_mae:.2f}°,   Standard: {std_mae:.2f}°")
    print(f"  RMSE Ang Err  -> Agent: {agent_rmse:.2f}°,   Standard: {std_rmse:.2f}°")
    print("=" * 60)
    
    # Determine success
    # PSNR: higher is better (allow 10% margin)
    # Correlation: higher is better (allow 10% margin)
    # Angular error: lower is better (allow 10% margin)
    
    success = True
    
    # PSNR check (higher is better)
    if std_psnr > 0 and agent_psnr < std_psnr * 0.9:
        print(f"FAIL: Agent PSNR ({agent_psnr:.2f}) significantly lower than Standard ({std_psnr:.2f})")
        success = False
    
    # Correlation check (higher is better)
    if std_cc > 0 and agent_cc < std_cc * 0.9:
        print(f"FAIL: Agent CC ({agent_cc:.4f}) significantly lower than Standard ({std_cc:.4f})")
        success = False
    
    # Mean angular error check (lower is better)
    if std_mae > 0 and agent_mae > std_mae * 1.5:
        print(f"FAIL: Agent MAE ({agent_mae:.2f}°) significantly higher than Standard ({std_mae:.2f}°)")
        success = False
    elif std_mae == 0 and agent_mae > 5.0:
        print(f"FAIL: Agent MAE ({agent_mae:.2f}°) too high (standard was near-perfect)")
        success = False
    
    if success:
        print("\nPASS: Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("\nFAIL: Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)