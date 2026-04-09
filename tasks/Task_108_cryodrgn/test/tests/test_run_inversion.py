import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from agent_run_inversion import run_inversion


def evaluate_results(ground_truth, reconstruction, projections, results_dir, script_dir,
                     n_projections, vol_size, noise_std):
    """
    Evaluate reconstruction quality and generate outputs.
    
    Metrics computed: PSNR, SSIM, RMSE
    """
    # Normalize volumes for metric computation
    gt_n = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-10)
    rec_n = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-10)
    
    # Compute RMSE
    rmse = float(np.sqrt(np.mean((gt_n - rec_n) ** 2)))
    
    # Compute per-axis SSIM and PSNR
    N = ground_truth.shape[0]
    mid = N // 2
    slices = [
        (gt_n[mid, :, :], rec_n[mid, :, :]),
        (gt_n[:, mid, :], rec_n[:, mid, :]),
        (gt_n[:, :, mid], rec_n[:, :, mid])
    ]
    ss = [float(ssim_metric(g, r, data_range=1.0)) for g, r in slices]
    ps = [float(psnr_metric(g, r, data_range=1.0)) for g, r in slices]
    
    # Compute 3D PSNR
    psnr_3d = float(psnr_metric(gt_n, rec_n, data_range=1.0))
    
    metrics = {
        'PSNR_dB': round(psnr_3d, 4),
        'SSIM': round(float(np.mean(ss)), 4),
        'RMSE': round(rmse, 6),
        'PSNR_per_axis': [round(v, 4) for v in ps],
        'SSIM_per_axis': [round(v, 4) for v in ss],
        'n_projections': n_projections,
        'volume_size': vol_size,
        'noise_std': noise_std,
    }
    
    print(f"  PSNR: {metrics['PSNR_dB']:.2f} dB")
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    
    # Save metrics and data
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), reconstruction)
    
    # Save normalized outputs
    np.save(os.path.join(script_dir, 'gt_output.npy'), gt_n)
    np.save(os.path.join(script_dir, 'recon_output.npy'), rec_n)
    print(f"  Saved gt_output.npy: range [{gt_n.min():.4f}, {gt_n.max():.4f}]")
    print(f"  Saved recon_output.npy: range [{rec_n.min():.4f}, {rec_n.max():.4f}]")
    
    # Generate visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # Ground truth slices
    axes[0, 0].imshow(gt_n[mid, :, :], cmap='gray')
    axes[0, 0].set_title('GT: Axial')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gt_n[:, mid, :], cmap='gray')
    axes[0, 1].set_title('GT: Coronal')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_n[:, :, mid], cmap='gray')
    axes[0, 2].set_title('GT: Sagittal')
    axes[0, 2].axis('off')
    axes[0, 3].imshow(np.max(gt_n, axis=0), cmap='hot')
    axes[0, 3].set_title('GT: MIP')
    axes[0, 3].axis('off')
    
    # Sample projections
    for j in range(4):
        idx = j * (len(projections) // 4)
        axes[1, j].imshow(projections[idx], cmap='gray')
        axes[1, j].set_title(f'Proj #{idx}')
        axes[1, j].axis('off')
    
    # Reconstruction slices
    axes[2, 0].imshow(rec_n[mid, :, :], cmap='gray')
    axes[2, 0].set_title('Recon: Axial')
    axes[2, 0].axis('off')
    axes[2, 1].imshow(rec_n[:, mid, :], cmap='gray')
    axes[2, 1].set_title('Recon: Coronal')
    axes[2, 1].axis('off')
    axes[2, 2].imshow(rec_n[:, :, mid], cmap='gray')
    axes[2, 2].set_title('Recon: Sagittal')
    axes[2, 2].axis('off')
    axes[2, 3].imshow(np.max(rec_n, axis=0), cmap='hot')
    axes[2, 3].set_title('Recon: MIP')
    axes[2, 3].axis('off')
    
    # Error maps
    for j, (title, sl) in enumerate([('Axial', mid), ('Coronal', mid), ('Sagittal', mid)]):
        if j == 0:
            err = np.abs(gt_n[sl, :, :] - rec_n[sl, :, :])
        elif j == 1:
            err = np.abs(gt_n[:, sl, :] - rec_n[:, sl, :])
        else:
            err = np.abs(gt_n[:, :, sl] - rec_n[:, :, sl])
        im = axes[3, j].imshow(err, cmap='hot', vmin=0, vmax=0.5)
        axes[3, j].set_title(f'Error: {title}')
        axes[3, j].axis('off')
        plt.colorbar(im, ax=axes[3, j], fraction=0.046)
    
    # Metrics text
    axes[3, 3].axis('off')
    t = (f"PSNR: {metrics['PSNR_dB']:.2f} dB\n"
         f"SSIM: {metrics['SSIM']:.4f}\n"
         f"RMSE: {metrics['RMSE']:.6f}\n"
         f"N_proj: {metrics['n_projections']}\n"
         f"Vol: {metrics['volume_size']}^3\n"
         f"Noise: {metrics['noise_std']}")
    axes[3, 3].text(0.1, 0.5, t, transform=axes[3, 3].transAxes, fontsize=14,
                    va='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3, 3].set_title('Metrics')
    
    fig.suptitle('Task 118: CryoDRGN - Cryo-EM 3D Reconstruction\n'
                 'Forward: Fourier Slice + CTF + Noise | Inverse: CTF-Weighted DFI',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    data_paths = ['/data/yjh/cryodrgn_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir_agent = os.path.join(script_dir, 'results_agent')
    results_dir_std = os.path.join(script_dir, 'results_std')
    script_dir_agent = os.path.join(script_dir, 'outputs_agent')
    script_dir_std = os.path.join(script_dir, 'outputs_std')
    
    os.makedirs(results_dir_agent, exist_ok=True)
    os.makedirs(results_dir_std, exist_ok=True)
    os.makedirs(script_dir_agent, exist_ok=True)
    os.makedirs(script_dir_std, exist_ok=True)
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)
    
    is_chained = len(inner_files) > 0
    
    print(f"Execution pattern: {'Chained' if is_chained else 'Direct'}")
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Load outer (primary) data
    outer_path = outer_files[0]
    print(f"\nLoading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Extract ground truth volume from the gen_data context
    # The evaluate_results function needs a ground_truth volume.
    # We need to figure out what the ground truth is. Looking at the gen_data code,
    # it captures inputs/outputs. The std_output IS the reconstruction from the standard code.
    # We need the ground truth volume separately.
    # 
    # Since evaluate_results compares ground_truth vs reconstruction, and we're testing
    # the agent's run_inversion against the standard run_inversion, we use the standard
    # output as the "ground truth" reference, and compare the agent's output against it.
    
    # Run agent's run_inversion
    print("\n=== Running Agent's run_inversion ===")
    try:
        agent_output = run_inversion(*args, **kwargs)
        print(f"Agent output shape: {agent_output.shape}")
        print(f"Agent output dtype: {agent_output.dtype}")
        print(f"Agent output range: [{agent_output.min():.6f}, {agent_output.max():.6f}]")
    except Exception as e:
        print(f"Agent run_inversion FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if is_chained:
        # Chained execution - run inner data through the operator
        inner_path = inner_files[0]
        print(f"\nLoading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print("Running agent operator on inner data...")
        try:
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Agent operator call FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution
        final_agent_result = agent_output
        std_result = std_output
    
    print(f"\nStandard result shape: {std_result.shape}")
    print(f"Standard result dtype: {std_result.dtype}")
    print(f"Standard result range: [{std_result.min():.6f}, {std_result.max():.6f}]")
    
    # Extract parameters for evaluate_results
    # From the run_inversion signature: projections, rot_mats, ctf_defocus, pixel_size, wiener_eps, apply_ctf
    projections = args[0] if len(args) > 0 else kwargs.get('projections')
    n_projections = len(projections)
    vol_size = projections.shape[1]  # N
    
    # Try to determine noise_std from context, default to 0.0 if not available
    noise_std = 0.0
    
    # Use the standard result as ground truth for evaluation
    # This way we measure how close the agent's output is to the reference implementation
    print("\n=== Evaluating Agent Output (against Standard as Ground Truth) ===")
    try:
        metrics_agent = evaluate_results(
            ground_truth=std_result,
            reconstruction=final_agent_result,
            projections=projections,
            results_dir=results_dir_agent,
            script_dir=script_dir_agent,
            n_projections=n_projections,
            vol_size=vol_size,
            noise_std=noise_std
        )
    except Exception as e:
        print(f"Agent evaluation FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Also evaluate standard against itself (should be perfect)
    print("\n=== Evaluating Standard Output (self-comparison baseline) ===")
    try:
        metrics_std = evaluate_results(
            ground_truth=std_result,
            reconstruction=std_result,
            projections=projections,
            results_dir=results_dir_std,
            script_dir=script_dir_std,
            n_projections=n_projections,
            vol_size=vol_size,
            noise_std=noise_std
        )
    except Exception as e:
        print(f"Standard evaluation FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics
    agent_psnr = metrics_agent['PSNR_dB']
    agent_ssim = metrics_agent['SSIM']
    agent_rmse = metrics_agent['RMSE']
    
    std_psnr = metrics_std['PSNR_dB']
    std_ssim = metrics_std['SSIM']
    std_rmse = metrics_std['RMSE']
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Agent  -> PSNR: {agent_psnr:.2f} dB, SSIM: {agent_ssim:.4f}, RMSE: {agent_rmse:.6f}")
    print(f"Std    -> PSNR: {std_psnr:.2f} dB, SSIM: {std_ssim:.4f}, RMSE: {std_rmse:.6f}")
    
    # For the agent vs standard comparison:
    # - PSNR: Higher is better. The standard self-comparison gives inf or very high PSNR.
    #   We need a reasonable threshold for the agent.
    # - SSIM: Higher is better (max 1.0). Standard self-comparison gives 1.0.
    # - RMSE: Lower is better. Standard self-comparison gives 0.0.
    
    # Since we're comparing agent output to the standard output:
    # A good agent should produce nearly identical results.
    # We'll set thresholds:
    #   - PSNR >= 25 dB (reasonable reconstruction quality)
    #   - SSIM >= 0.85
    #   - RMSE <= 0.1
    
    # Also do a direct numpy comparison
    direct_diff = np.sqrt(np.mean((final_agent_result - std_result) ** 2))
    print(f"\nDirect RMSE (agent vs std): {direct_diff:.6f}")
    
    # Determine pass/fail
    passed = True
    reasons = []
    
    # If PSNR is very high (close to identical), that's great
    if agent_psnr < 25.0:
        reasons.append(f"PSNR too low: {agent_psnr:.2f} dB < 25.0 dB")
        passed = False
    
    if agent_ssim < 0.85:
        reasons.append(f"SSIM too low: {agent_ssim:.4f} < 0.85")
        passed = False
    
    if agent_rmse > 0.1:
        reasons.append(f"RMSE too high: {agent_rmse:.6f} > 0.1")
        passed = False
    
    print("\n" + "=" * 60)
    if passed:
        print("TEST PASSED: Agent reconstruction matches standard output.")
        print(f"  PSNR: {agent_psnr:.2f} dB (threshold: >= 25.0)")
        print(f"  SSIM: {agent_ssim:.4f} (threshold: >= 0.85)")
        print(f"  RMSE: {agent_rmse:.6f} (threshold: <= 0.1)")
        sys.exit(0)
    else:
        print("TEST FAILED: Agent reconstruction does NOT match standard output.")
        for r in reasons:
            print(f"  FAIL: {r}")
        sys.exit(1)


if __name__ == '__main__':
    main()