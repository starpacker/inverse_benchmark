import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity

sys.path.insert(0, '/data/yjh/odtbrain_sandbox/repo')

RESULTS_DIR = '/data/yjh/odtbrain_sandbox/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import the target function
from agent_run_inversion import run_inversion

# Inject the evaluate_results function verbatim from Reference B
def evaluate_results(phantom_ri, ri_recon_aligned, sino_noisy, params, results_dir):
    """
    Compute metrics, create visualizations, and save results.
    """
    N = params['N']
    nm = params['nm']
    n_sphere = params['n_sphere']
    num_angles = params['num_angles']
    noise_level = params['noise_level']

    def compute_psnr(gt, recon):
        mse = np.mean((gt - recon) ** 2)
        if mse == 0:
            return float('inf')
        data_range = gt.max() - gt.min()
        return 10 * np.log10(data_range ** 2 / mse)

    def compute_ssim(gt, recon, data_range=None):
        if data_range is None:
            data_range = gt.max() - gt.min()
        return structural_similarity(gt, recon, data_range=data_range)

    def compute_rmse(gt, recon):
        return np.sqrt(np.mean((gt - recon) ** 2))

    center = N // 2
    gt_slice = phantom_ri[center, :, :]
    recon_slice = ri_recon_aligned[center, :, :]

    data_range = phantom_ri.max() - phantom_ri.min()

    psnr_2d = compute_psnr(gt_slice, recon_slice)
    ssim_2d = compute_ssim(gt_slice, recon_slice, data_range=data_range)
    rmse_2d = compute_rmse(gt_slice, recon_slice)

    psnr_3d = compute_psnr(phantom_ri, ri_recon_aligned)
    ssim_slices = []
    for i in range(phantom_ri.shape[0]):
        s = compute_ssim(phantom_ri[i], ri_recon_aligned[i], data_range=data_range)
        ssim_slices.append(s)
    ssim_3d = float(np.mean(ssim_slices))
    rmse_3d = compute_rmse(phantom_ri, ri_recon_aligned)

    print(f"  Central slice - PSNR: {psnr_2d:.2f} dB, SSIM: {ssim_2d:.4f}, RMSE: {rmse_2d:.6f}")
    print(f"  3D volume     - PSNR: {psnr_3d:.2f} dB, SSIM: {ssim_3d:.4f}, RMSE: {rmse_3d:.6f}")

    metrics = {
        'PSNR': round(float(psnr_2d), 2),
        'SSIM': round(float(ssim_2d), 4),
        'RMSE': round(float(rmse_2d), 6),
        'PSNR_3D': round(float(psnr_3d), 2),
        'SSIM_3D': round(float(ssim_3d), 4),
        'RMSE_3D': round(float(rmse_3d), 6),
        'num_projections': num_angles,
        'grid_size': N,
        'noise_level': noise_level,
        'medium_index': nm,
        'sphere_index': n_sphere,
        'method': 'Rytov backpropagation (ODTbrain)'
    }

    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(gt_slice, vmin=nm-0.001, vmax=n_sphere+0.001,
                              cmap='hot', interpolation='none')
    axes[0, 0].set_title('Ground Truth (central xz slice)', fontsize=11)
    axes[0, 0].set_xlabel('x [px]')
    axes[0, 0].set_ylabel('z [px]')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, format='%.4f')

    sino_phase = np.angle(sino_noisy)
    im1 = axes[0, 1].imshow(sino_phase[:, center, :],
                              aspect=sino_noisy.shape[2]/sino_noisy.shape[0],
                              cmap='coolwarm', interpolation='none')
    axes[0, 1].set_title('Phase Sinogram (y=center)', fontsize=11)
    axes[0, 1].set_xlabel('detector x [px]')
    axes[0, 1].set_ylabel('angle index')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(recon_slice, vmin=nm-0.001, vmax=n_sphere+0.001,
                              cmap='hot', interpolation='none')
    axes[0, 2].set_title(f'Reconstruction (PSNR={psnr_2d:.1f}dB, SSIM={ssim_2d:.3f})',
                          fontsize=11)
    axes[0, 2].set_xlabel('x [px]')
    axes[0, 2].set_ylabel('z [px]')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, format='%.4f')

    error = np.abs(gt_slice - recon_slice)
    im3 = axes[1, 0].imshow(error, cmap='hot', interpolation='none')
    axes[1, 0].set_title('Absolute Error Map', fontsize=11)
    axes[1, 0].set_xlabel('x [px]')
    axes[1, 0].set_ylabel('z [px]')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, format='%.5f')

    center_line_gt = phantom_ri[center, center, :]
    center_line_recon = ri_recon_aligned[center, center, :]
    axes[1, 1].plot(center_line_gt, 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(center_line_recon, 'r--', linewidth=2, label='Reconstruction')
    axes[1, 1].set_title('Line Profile (through center)', fontsize=11)
    axes[1, 1].set_xlabel('x [px]')
    axes[1, 1].set_ylabel('Refractive Index')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([nm - 0.002, n_sphere + 0.002])

    im5 = axes[1, 2].imshow(ri_recon_aligned[:, :, center].T,
                              vmin=nm-0.001, vmax=n_sphere+0.001,
                              cmap='hot', interpolation='none')
    axes[1, 2].set_title('Reconstruction (yz slice at x=center)', fontsize=11)
    axes[1, 2].set_xlabel('z [px]')
    axes[1, 2].set_ylabel('y [px]')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, format='%.4f')

    plt.suptitle('ODTbrain: 3D RI Reconstruction via Rytov Backpropagation\n'
                 f'N={N}, {num_angles} projections, noise={noise_level}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {fig_path}")

    gt_path = os.path.join(results_dir, 'ground_truth.npy')
    recon_path = os.path.join(results_dir, 'reconstruction.npy')
    np.save(gt_path, phantom_ri)
    np.save(recon_path, ri_recon_aligned)
    print(f"  Ground truth saved to {gt_path}")
    print(f"  Reconstruction saved to {recon_path}")

    return metrics


def main():
    data_paths = ['/data/yjh/odtbrain_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer and inner data files
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

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}, Number of kwargs: {len(kwargs)}")

    # Run the agent's function
    print("\nRunning agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Chained execution pattern
        print(f"\nDetected chained execution with {len(inner_paths)} inner file(s).")
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Inner call failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        print("\nDirect execution pattern detected.")
        final_result = agent_output
        std_result = std_output

    # Both final_result and std_result should be dicts with 'ri_recon_aligned', etc.
    print("\n--- Validating outputs ---")
    print(f"Agent output type: {type(final_result)}")
    print(f"Standard output type: {type(std_result)}")

    if isinstance(final_result, dict):
        print(f"Agent output keys: {list(final_result.keys())}")
    if isinstance(std_result, dict):
        print(f"Standard output keys: {list(std_result.keys())}")

    # We need phantom_ri and params for evaluate_results
    # Extract sino_noisy from the input args
    # From the function signature: run_inversion(sino_noisy, angles, res, nm, lD, phantom_shape)
    sino_noisy = args[0] if len(args) > 0 else kwargs.get('sino_noisy')
    angles = args[1] if len(args) > 1 else kwargs.get('angles')
    res = args[2] if len(args) > 2 else kwargs.get('res')
    nm = args[3] if len(args) > 3 else kwargs.get('nm')
    lD = args[4] if len(args) > 4 else kwargs.get('lD')
    phantom_shape = args[5] if len(args) > 5 else kwargs.get('phantom_shape')

    print(f"\nPhantom shape: {phantom_shape}")
    print(f"nm: {nm}, res: {res}, lD: {lD}")
    print(f"Number of angles: {len(angles)}")
    print(f"sino_noisy shape: {sino_noisy.shape}")

    # We need phantom_ri for evaluation. We'll reconstruct it from the standard output
    # by using the standard result as a reference and comparing metrics relatively.
    # However, evaluate_results needs the ground truth phantom_ri.
    # 
    # Since we don't have phantom_ri directly, we need to create a simple phantom.
    # But wait - the standard data should contain the ground truth implicitly.
    # Let's check if there's a ground truth file or if we can derive params.
    
    # For evaluation, we use the standard result's ri_recon_aligned as a proxy ground truth
    # to compare against the agent's result. This tests that the agent produces
    # equivalent results to the standard.
    
    # Actually, the proper approach: we compare agent vs standard outputs directly
    # using numerical metrics, since we don't have the original phantom.
    
    # Let's compute direct comparison metrics between agent and standard outputs
    agent_ri = final_result['ri_recon_aligned']
    std_ri = std_result['ri_recon_aligned']
    
    print(f"\nAgent ri_recon_aligned shape: {agent_ri.shape}, dtype: {agent_ri.dtype}")
    print(f"Standard ri_recon_aligned shape: {std_ri.shape}, dtype: {std_ri.dtype}")

    # Also try to use evaluate_results by treating std as ground truth
    # Build params dict
    N = phantom_shape[0]  # Assume cubic phantom
    num_angles = len(angles)
    
    # Estimate n_sphere from the standard reconstruction
    n_sphere_est = float(np.max(std_ri))
    noise_level_est = 0.0  # We don't know the exact noise level, use a placeholder

    params_agent = {
        'N': N,
        'nm': nm,
        'n_sphere': n_sphere_est,
        'num_angles': num_angles,
        'noise_level': noise_level_est
    }

    # Evaluate agent output against standard output (treating std as ground truth)
    print("\n=== Evaluating Agent output (vs Standard as reference) ===")
    agent_results_dir = os.path.join(RESULTS_DIR, 'agent_eval')
    os.makedirs(agent_results_dir, exist_ok=True)
    
    try:
        metrics_agent = evaluate_results(
            phantom_ri=std_ri,
            ri_recon_aligned=agent_ri,
            sino_noisy=sino_noisy,
            params=params_agent,
            results_dir=agent_results_dir
        )
    except Exception as e:
        print(f"ERROR in agent evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Evaluate standard output against itself (perfect score baseline)
    print("\n=== Evaluating Standard output (vs itself, baseline) ===")
    std_results_dir = os.path.join(RESULTS_DIR, 'std_eval')
    os.makedirs(std_results_dir, exist_ok=True)
    
    try:
        metrics_std = evaluate_results(
            phantom_ri=std_ri,
            ri_recon_aligned=std_ri,
            sino_noisy=sino_noisy,
            params=params_agent,
            results_dir=std_results_dir
        )
    except Exception as e:
        print(f"ERROR in standard evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract key metrics
    psnr_agent = metrics_agent['PSNR']
    ssim_agent = metrics_agent['SSIM']
    rmse_agent = metrics_agent['RMSE']
    psnr_3d_agent = metrics_agent['PSNR_3D']
    ssim_3d_agent = metrics_agent['SSIM_3D']
    rmse_3d_agent = metrics_agent['RMSE_3D']

    psnr_std = metrics_std['PSNR']
    ssim_std = metrics_std['SSIM']
    rmse_std = metrics_std['RMSE']

    print(f"\n{'='*60}")
    print(f"Scores -> Agent PSNR: {psnr_agent}, Standard PSNR: {psnr_std}")
    print(f"Scores -> Agent SSIM: {ssim_agent}, Standard SSIM: {ssim_std}")
    print(f"Scores -> Agent RMSE: {rmse_agent}, Standard RMSE: {rmse_std}")
    print(f"Scores -> Agent PSNR_3D: {psnr_3d_agent}, Agent SSIM_3D: {ssim_3d_agent}")
    print(f"{'='*60}")

    # Additional direct comparison
    max_abs_diff = float(np.max(np.abs(agent_ri - std_ri)))
    mean_abs_diff = float(np.mean(np.abs(agent_ri - std_ri)))
    print(f"\nDirect comparison - Max abs diff: {max_abs_diff:.10f}")
    print(f"Direct comparison - Mean abs diff: {mean_abs_diff:.10f}")

    # Also compare f_recon and sino_rytov
    agent_f = final_result['f_recon']
    std_f = std_result['f_recon']
    f_max_diff = float(np.max(np.abs(agent_f - std_f)))
    f_mean_diff = float(np.mean(np.abs(agent_f - std_f)))
    print(f"f_recon comparison - Max abs diff: {f_max_diff:.10e}")
    print(f"f_recon comparison - Mean abs diff: {f_mean_diff:.10e}")

    agent_sino = final_result['sino_rytov']
    std_sino = std_result['sino_rytov']
    sino_max_diff = float(np.max(np.abs(agent_sino - std_sino)))
    sino_mean_diff = float(np.mean(np.abs(agent_sino - std_sino)))
    print(f"sino_rytov comparison - Max abs diff: {sino_max_diff:.10e}")
    print(f"sino_rytov comparison - Mean abs diff: {sino_mean_diff:.10e}")

    # Verification logic
    # When comparing agent to standard (as ground truth):
    # - PSNR should be very high (close to inf for identical results)
    # - SSIM should be very close to 1.0
    # - RMSE should be very close to 0
    # 
    # We use lenient thresholds since numerical differences can arise
    # from floating point, threading, etc.
    
    passed = True
    
    # PSNR: Higher is better. Agent compared to std should have high PSNR.
    # If results are very similar, PSNR should be > 40 dB at minimum.
    # For near-identical results, it could be inf or very high.
    min_psnr_threshold = 30.0  # Very lenient threshold
    if psnr_agent < min_psnr_threshold and psnr_agent != float('inf'):
        print(f"FAIL: Agent PSNR ({psnr_agent}) < threshold ({min_psnr_threshold})")
        passed = False
    else:
        print(f"PASS: Agent PSNR ({psnr_agent}) >= threshold ({min_psnr_threshold})")

    # SSIM: Should be very close to 1.0
    min_ssim_threshold = 0.90
    if ssim_agent < min_ssim_threshold:
        print(f"FAIL: Agent SSIM ({ssim_agent}) < threshold ({min_ssim_threshold})")
        passed = False
    else:
        print(f"PASS: Agent SSIM ({ssim_agent}) >= threshold ({min_ssim_threshold})")

    # SSIM 3D
    min_ssim_3d_threshold = 0.90
    if ssim_3d_agent < min_ssim_3d_threshold:
        print(f"FAIL: Agent SSIM_3D ({ssim_3d_agent}) < threshold ({min_ssim_3d_threshold})")
        passed = False
    else:
        print(f"PASS: Agent SSIM_3D ({ssim_3d_agent}) >= threshold ({min_ssim_3d_threshold})")

    if passed:
        print("\n*** ALL CHECKS PASSED ***")
        sys.exit(0)
    else:
        print("\n*** SOME CHECKS FAILED ***")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)