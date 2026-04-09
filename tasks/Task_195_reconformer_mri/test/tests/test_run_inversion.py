import sys
import os
import dill
import numpy as np
import traceback
import json

# Import matplotlib with Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import metrics from skimage
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Import the target function
from agent_run_inversion import run_inversion

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the evaluate_results function (Reference B)
def evaluate_results(data, inversion_result):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR, SSIM, RMSE metrics for:
      - Zero-filled baseline
      - Raw reconstruction
      - Intensity-corrected reconstruction
    
    Saves:
      - metrics.json
      - ground_truth.npy, reconstruction.npy, zero_filled.npy, mask.npy
      - reconstruction_result.png visualization
    
    Args:
        data: Dictionary from load_and_preprocess_data
        inversion_result: Dictionary from run_inversion
    
    Returns:
        dict: All computed metrics
    """
    gt_image = data['gt_image']
    zero_filled = data['zero_filled']
    mask = data['mask']
    
    recon_raw = inversion_result['recon_raw']
    final_recon = inversion_result['final_recon']
    
    def compute_metrics(gt, recon):
        data_range = gt.max() - gt.min() + 1e-12
        gt_norm = (gt - gt.min()) / data_range
        recon_norm = np.clip((recon - gt.min()) / data_range, 0, 1)
        
        p = float(psnr_metric(gt_norm, recon_norm, data_range=1.0))
        s = float(ssim_metric(gt_norm, recon_norm, data_range=1.0))
        r = float(np.sqrt(np.mean((gt_norm - recon_norm)**2)))
        
        return {'psnr': round(p, 4), 'ssim': round(s, 4), 'rmse': round(r, 6)}
    
    metrics_zf = compute_metrics(gt_image, zero_filled)
    metrics_raw = compute_metrics(gt_image, recon_raw)
    metrics_corrected = compute_metrics(gt_image, final_recon)
    
    print(f"\n  Zero-filled baseline: PSNR={metrics_zf['psnr']:.2f} dB, "
          f"SSIM={metrics_zf['ssim']:.4f}")
    print(f"  Raw reconstruction: PSNR={metrics_raw['psnr']:.2f} dB, "
          f"SSIM={metrics_raw['ssim']:.4f}")
    print(f"  Corrected: PSNR={metrics_corrected['psnr']:.2f} dB, "
          f"SSIM={metrics_corrected['ssim']:.4f}")
    
    # Get N from data if available, otherwise infer from gt_image shape
    N = data.get('N', gt_image.shape[0])
    
    all_metrics = {
        'task': 'reconformer_mri',
        'task_id': 195,
        'method': 'FISTA + TV (Compressed Sensing MRI)',
        'acceleration': 4,
        'image_size': N,
        'sampling_rate': 0.25,
        'tv_lambda': 0.0003,
        'fista_iterations': 1300,
        'zero_filled': metrics_zf,
        'raw_reconstruction': metrics_raw,
        'corrected_reconstruction': metrics_corrected,
        'psnr': metrics_corrected['psnr'],
        'ssim': metrics_corrected['ssim'],
        'rmse': metrics_corrected['rmse'],
    }
    
    print("\n  Saving results...")
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")
    
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), gt_image)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), final_recon)
    np.save(os.path.join(RESULTS_DIR, 'zero_filled.npy'), zero_filled)
    np.save(os.path.join(RESULTS_DIR, 'mask.npy'), mask)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = gt_image.min(), gt_image.max()
    
    axes[0].imshow(gt_image, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('(a) Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(zero_filled, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'(b) Zero-filled\nPSNR={metrics_zf["psnr"]:.2f} dB, '
                      f'SSIM={metrics_zf["ssim"]:.4f}', fontsize=11)
    axes[1].axis('off')
    
    axes[2].imshow(final_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'(c) CS-TV Reconstruction\nPSNR={metrics_corrected["psnr"]:.2f} dB, '
                      f'SSIM={metrics_corrected["ssim"]:.4f}', fontsize=11)
    axes[2].axis('off')
    
    error = np.abs(gt_image - final_recon)
    im = axes[3].imshow(error, cmap='hot', vmin=0, vmax=max(error.max() * 0.5, 1e-6))
    axes[3].set_title('(d) Error Map |GT - Recon|', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    fig.suptitle('Accelerated MRI Reconstruction (4x Cartesian Undersampling)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    vis_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {vis_path}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Zero-filled:   PSNR={metrics_zf['psnr']:.2f} dB, SSIM={metrics_zf['ssim']:.4f}")
    print(f"  Raw CS-TV:     PSNR={metrics_raw['psnr']:.2f} dB, SSIM={metrics_raw['ssim']:.4f}")
    print(f"  Final (corr.): PSNR={metrics_corrected['psnr']:.2f} dB, SSIM={metrics_corrected['ssim']:.4f}")
    print(f"  RMSE:          {metrics_corrected['rmse']:.6f}")
    
    target_met = metrics_corrected['psnr'] > 25 and metrics_corrected['ssim'] > 0.85
    print(f"\n  Target (PSNR>25, SSIM>0.85): {'MET ✓' if target_met else 'NOT MET ✗'}")
    print("=" * 70)
    
    return all_metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/reconformer_mri_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 70)
    print("TEST: run_inversion Performance Validation")
    print("=" * 70)
    
    # Analyze data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"\nOuter data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    print(f"\nLoading outer data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"  Loaded successfully. Keys: {list(outer_data.keys()) if isinstance(outer_data, dict) else 'N/A'}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"\n  Args count: {len(args)}")
    print(f"  Kwargs keys: {list(kwargs.keys())}")
    
    # Execute the agent's run_inversion
    print("\n" + "-" * 70)
    print("EXECUTING AGENT'S run_inversion")
    print("-" * 70)
    
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("\n  Agent execution completed successfully.")
    except Exception as e:
        print(f"ERROR during agent execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if chained execution is needed
    if len(inner_data_paths) > 0:
        # Chained execution pattern
        print("\n" + "-" * 70)
        print("CHAINED EXECUTION DETECTED")
        print("-" * 70)
        
        for inner_path in inner_data_paths:
            print(f"\nLoading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
    else:
        # Direct execution pattern
        final_result = agent_output
        std_result = std_output
    
    # Get the input data for evaluation
    # The first argument should be the 'data' dictionary
    if len(args) > 0:
        input_data = args[0]
    else:
        input_data = kwargs.get('data', None)
    
    if input_data is None:
        print("ERROR: Could not extract input data for evaluation!")
        sys.exit(1)
    
    # Ensure input_data has required fields for evaluation
    # Check if 'zero_filled' exists, if not create it from y_kspace
    if 'zero_filled' not in input_data:
        if 'y_kspace' in input_data:
            input_data['zero_filled'] = np.real(np.fft.ifft2(input_data['y_kspace'], norm='ortho'))
            print("  Created 'zero_filled' from y_kspace")
    
    if 'N' not in input_data:
        if 'gt_image' in input_data:
            input_data['N'] = input_data['gt_image'].shape[0]
    
    # Evaluate agent's result
    print("\n" + "-" * 70)
    print("EVALUATING AGENT'S RESULT")
    print("-" * 70)
    
    try:
        agent_metrics = evaluate_results(input_data, final_result)
        agent_psnr = agent_metrics['psnr']
        agent_ssim = agent_metrics['ssim']
        agent_rmse = agent_metrics['rmse']
        print(f"\n  Agent PSNR: {agent_psnr:.4f} dB")
        print(f"  Agent SSIM: {agent_ssim:.4f}")
        print(f"  Agent RMSE: {agent_rmse:.6f}")
    except Exception as e:
        print(f"ERROR during agent evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("\n" + "-" * 70)
    print("EVALUATING STANDARD RESULT")
    print("-" * 70)
    
    try:
        # Update results directory for standard evaluation
        std_results_dir = os.path.join(WORKING_DIR, "results_std")
        os.makedirs(std_results_dir, exist_ok=True)
        
        # Temporarily change RESULTS_DIR for standard evaluation
        global RESULTS_DIR
        original_results_dir = RESULTS_DIR
        RESULTS_DIR = std_results_dir
        
        std_metrics = evaluate_results(input_data, std_result)
        std_psnr = std_metrics['psnr']
        std_ssim = std_metrics['ssim']
        std_rmse = std_metrics['rmse']
        
        # Restore RESULTS_DIR
        RESULTS_DIR = original_results_dir
        
        print(f"\n  Standard PSNR: {std_psnr:.4f} dB")
        print(f"  Standard SSIM: {std_ssim:.4f}")
        print(f"  Standard RMSE: {std_rmse:.6f}")
    except Exception as e:
        print(f"ERROR during standard evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\n  Scores -> Agent: PSNR={agent_psnr:.4f}, SSIM={agent_ssim:.4f}")
    print(f"  Scores -> Standard: PSNR={std_psnr:.4f}, SSIM={std_ssim:.4f}")
    
    # Determine success
    # For PSNR and SSIM, higher is better
    # Allow 10% margin of error
    psnr_threshold = std_psnr * 0.9
    ssim_threshold = std_ssim * 0.9
    
    psnr_pass = agent_psnr >= psnr_threshold
    ssim_pass = agent_ssim >= ssim_threshold
    
    print(f"\n  PSNR Check: Agent ({agent_psnr:.4f}) >= {psnr_threshold:.4f} (90% of std): {'PASS ✓' if psnr_pass else 'FAIL ✗'}")
    print(f"  SSIM Check: Agent ({agent_ssim:.4f}) >= {ssim_threshold:.4f} (90% of std): {'PASS ✓' if ssim_pass else 'FAIL ✗'}")
    
    # Additional check: meet minimum targets (PSNR > 25, SSIM > 0.85)
    target_psnr = agent_psnr > 25
    target_ssim = agent_ssim > 0.85
    
    print(f"\n  Target PSNR > 25: {agent_psnr:.4f} -> {'PASS ✓' if target_psnr else 'FAIL ✗'}")
    print(f"  Target SSIM > 0.85: {agent_ssim:.4f} -> {'PASS ✓' if target_ssim else 'FAIL ✗'}")
    
    print("\n" + "=" * 70)
    
    if psnr_pass and ssim_pass:
        print("TEST RESULT: PASSED ✓")
        print("Agent's performance is within acceptable range of standard.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("TEST RESULT: FAILED ✗")
        print("Agent's performance degraded significantly compared to standard.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()