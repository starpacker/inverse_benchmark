import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Import the target function
from agent_run_inversion import run_inversion


def evaluate_results(phantom, reconstruction, sinograms_noisy, theta, params, output_dir='results'):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, RMSE metrics for both the full 3D volume and middle slice.
    Saves metrics to JSON, arrays to NPY, and creates visualization.
    
    Args:
        phantom: 3D ground truth array
        reconstruction: 3D reconstructed array
        sinograms_noisy: 3D array of noisy sinograms
        theta: projection angles
        params: dictionary of parameters
        output_dir: directory to save results
    
    Returns:
        metrics: dictionary containing all computed metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    nz, ny, nx = phantom.shape
    mid_z = nz // 2
    
    # Compute 3D volume metrics
    data_range_3d = phantom.max() - phantom.min()
    if data_range_3d < 1e-10:
        data_range_3d = 1.0
    
    psnr_3d = peak_signal_noise_ratio(phantom, reconstruction, data_range=data_range_3d)
    ssim_3d = structural_similarity(phantom, reconstruction, data_range=data_range_3d)
    rmse_3d = np.sqrt(np.mean((phantom - reconstruction) ** 2))
    
    # Compute middle slice metrics
    gt_mid = phantom[mid_z]
    recon_mid = reconstruction[mid_z]
    data_range_mid = gt_mid.max() - gt_mid.min()
    if data_range_mid < 1e-10:
        data_range_mid = 1.0
    
    psnr_mid = peak_signal_noise_ratio(gt_mid, recon_mid, data_range=data_range_mid)
    ssim_mid = structural_similarity(gt_mid, recon_mid, data_range=data_range_mid)
    rmse_mid = np.sqrt(np.mean((gt_mid - recon_mid) ** 2))
    
    # Build metrics dictionary
    metrics = {
        'task': 'opt_recon',
        'task_number': 119,
        'method': 'Filtered Back-Projection (FBP)',
        'inverse_problem': 'Optical Projection Tomography (OPT) 3D Reconstruction',
        'phantom_shape': list(phantom.shape),
        'n_angles': params['n_angles'],
        'photon_count': params['photon_count'],
        'readout_noise_std': params['readout_std'],
        'metrics_3d': {
            'PSNR_dB': round(psnr_3d, 2),
            'SSIM': round(ssim_3d, 4),
            'RMSE': round(rmse_3d, 6)
        },
        'metrics_middle_slice': {
            'PSNR_dB': round(psnr_mid, 2),
            'SSIM': round(ssim_mid, 4),
            'RMSE': round(rmse_mid, 6)
        }
    }
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(output_dir, 'ground_truth.npy'), phantom)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), reconstruction)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Task 119: OPT Reconstruction (Filtered Back-Projection)',
                 fontsize=16, fontweight='bold')
    
    gt_slice = phantom[mid_z]
    sino_slice = sinograms_noisy[mid_z]
    recon_slice = reconstruction[mid_z]
    error_map = np.abs(gt_slice - recon_slice)
    
    vmin, vmax = 0, gt_slice.max()
    
    # GT slice
    im0 = axes[0, 0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Ground Truth (z={mid_z})', fontsize=13)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Sinogram
    im1 = axes[0, 1].imshow(sino_slice.T, cmap='gray', aspect='auto',
                             extent=[0, sino_slice.shape[0], 180, 0])
    axes[0, 1].set_title(f'Sinogram (noisy, z={mid_z})', fontsize=13)
    axes[0, 1].set_xlabel('Detector position')
    axes[0, 1].set_ylabel('Angle (degrees)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Reconstruction
    im2 = axes[0, 2].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'FBP Reconstruction (z={mid_z})', fontsize=13)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Error map
    im3 = axes[1, 0].imshow(error_map, cmap='hot')
    axes[1, 0].set_title('Absolute Error Map', fontsize=13)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Line profile comparison
    mid_y = ny // 2
    axes[1, 1].plot(gt_slice[mid_y, :], 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(recon_slice[mid_y, :], 'r--', linewidth=2, label='FBP Recon')
    axes[1, 1].set_title(f'Line Profile (y={mid_y})', fontsize=13)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3D overview: show multiple slices
    slice_indices = [nz // 4, nz // 2, 3 * nz // 4]
    for i, zi in enumerate(slice_indices):
        color = ['blue', 'green', 'red'][i]
        gt_zi = phantom[zi]
        recon_zi = reconstruction[zi]
        dr_zi = gt_zi.max() - gt_zi.min()
        if dr_zi < 1e-10:
            dr_zi = 1.0
        psnr_i = peak_signal_noise_ratio(gt_zi, recon_zi, data_range=dr_zi)
        axes[1, 2].plot(reconstruction[zi][ny // 2, :], color=color, alpha=0.7,
                        label=f'z={zi} (PSNR={psnr_i:.1f}dB)')
    axes[1, 2].set_title('Reconstruction Profiles at Different z-Slices', fontsize=13)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add metrics text
    fig.text(0.5, 0.01,
             f'3D Volume --- PSNR: {psnr_3d:.2f} dB | SSIM: {ssim_3d:.4f} | RMSE: {rmse_3d:.6f}',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    data_paths = ['/data/yjh/opt_recon_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

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

    print(f"Function: {func_name}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")

    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("Pattern 2: Chained Execution detected")
        print(f"Running outer function: run_inversion(*args, **kwargs)")
        try:
            operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running outer function: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        print(f"Running operator(*inner_args, **inner_kwargs)")
        try:
            agent_result = operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("Pattern 1: Direct Execution detected")
        print(f"Running: run_inversion(*args, **kwargs)")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running function: {e}")
            traceback.print_exc()
            sys.exit(1)
        std_result = std_output

    print(f"Agent result type: {type(agent_result)}, shape: {agent_result.shape if hasattr(agent_result, 'shape') else 'N/A'}")
    print(f"Std result type: {type(std_result)}, shape: {std_result.shape if hasattr(std_result, 'shape') else 'N/A'}")

    # Now we need to evaluate. evaluate_results requires phantom, reconstruction, sinograms_noisy, theta, params.
    # We need to extract these from the data. 
    # The function signature is: run_inversion(sinograms_noisy, theta, output_shape)
    # args[0] = sinograms_noisy, args[1] = theta, args[2] = output_shape
    
    sinograms_noisy = args[0] if len(args) > 0 else kwargs.get('sinograms_noisy')
    theta = args[1] if len(args) > 1 else kwargs.get('theta')
    output_shape = args[2] if len(args) > 2 else kwargs.get('output_shape')

    # We don't have the ground truth phantom directly from the inversion data.
    # For evaluation, we'll use the standard result as "phantom" for both evaluations,
    # or we compare agent vs standard directly.
    # 
    # Actually, evaluate_results compares reconstruction against phantom.
    # Since we don't have the true phantom, we'll use a direct comparison approach:
    # Use std_result as the reference "phantom" for both, OR compare metrics directly.
    #
    # Better approach: Compare agent_result vs std_result directly using PSNR/SSIM,
    # and also use std_result as phantom to evaluate agent quality.

    # Build params dict for evaluate_results
    n_angles = len(theta) if theta is not None else 0
    params = {
        'n_angles': n_angles,
        'photon_count': 1e5,  # default, not critical for metric computation
        'readout_std': 5.0,   # default, not critical for metric computation
    }

    # Evaluate agent result against standard result (use std as "phantom")
    print("\n--- Evaluating Agent Result against Standard Result ---")
    try:
        agent_metrics = evaluate_results(
            phantom=std_result,
            reconstruction=agent_result,
            sinograms_noisy=sinograms_noisy,
            theta=theta,
            params=params,
            output_dir='results_agent_vs_std'
        )
    except Exception as e:
        print(f"ERROR in evaluate_results for agent: {e}")
        traceback.print_exc()
        sys.exit(1)

    psnr_agent = agent_metrics['metrics_3d']['PSNR_dB']
    ssim_agent = agent_metrics['metrics_3d']['SSIM']
    rmse_agent = agent_metrics['metrics_3d']['RMSE']

    print(f"\nAgent vs Standard comparison:")
    print(f"  PSNR: {psnr_agent} dB")
    print(f"  SSIM: {ssim_agent}")
    print(f"  RMSE: {rmse_agent}")

    # Also check direct numerical closeness
    max_abs_diff = np.max(np.abs(agent_result - std_result))
    mean_abs_diff = np.mean(np.abs(agent_result - std_result))
    print(f"\nDirect comparison:")
    print(f"  Max absolute difference: {max_abs_diff}")
    print(f"  Mean absolute difference: {mean_abs_diff}")

    # Evaluate: if agent result is very close to standard, PSNR should be very high
    # For near-identical results, PSNR > 40 dB and SSIM > 0.99
    # We allow some margin: PSNR > 30 dB and SSIM > 0.95

    # Also evaluate standard result self-consistency (should be perfect)
    print("\n--- Evaluating Standard Result self-consistency ---")
    try:
        std_metrics = evaluate_results(
            phantom=std_result,
            reconstruction=std_result,
            sinograms_noisy=sinograms_noisy,
            theta=theta,
            params=params,
            output_dir='results_std_self'
        )
    except Exception as e:
        print(f"ERROR in evaluate_results for std: {e}")
        traceback.print_exc()
        sys.exit(1)

    psnr_std = std_metrics['metrics_3d']['PSNR_dB']
    ssim_std = std_metrics['metrics_3d']['SSIM']
    print(f"\nStandard self-comparison (should be perfect):")
    print(f"  PSNR: {psnr_std} dB")
    print(f"  SSIM: {ssim_std}")

    print(f"\n=== Final Scores ===")
    print(f"Scores -> Agent vs Std PSNR: {psnr_agent} dB, SSIM: {ssim_agent}, RMSE: {rmse_agent}")

    # Decision: agent output should be very close to standard output
    # For FBP which is deterministic, we expect near-perfect match
    # Use generous thresholds: PSNR > 30 dB and SSIM > 0.90
    passed = True
    
    if psnr_agent < 30.0:
        print(f"FAIL: PSNR {psnr_agent} dB < 30.0 dB threshold")
        passed = False
    else:
        print(f"PASS: PSNR {psnr_agent} dB >= 30.0 dB threshold")

    if ssim_agent < 0.90:
        print(f"FAIL: SSIM {ssim_agent} < 0.90 threshold")
        passed = False
    else:
        print(f"PASS: SSIM {ssim_agent} >= 0.90 threshold")

    if passed:
        print("\n=== TEST PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TEST FAILED ===")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)