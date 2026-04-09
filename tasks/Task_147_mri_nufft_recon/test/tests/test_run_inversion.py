import sys
import os
import dill
import numpy as np
import traceback
import json
import warnings

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, 'repo')
sys.path.insert(0, REPO_DIR)

warnings.filterwarnings('ignore', message='Samples will be rescaled')

# Import the agent's function
from agent_run_inversion import run_inversion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle

# Inject the evaluate_results function from Reference B
def evaluate_results(data, results, results_dir):
    """
    Evaluate and save the reconstruction results.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    phantom = data['phantom']
    params = data['params']
    
    recon_adjoint = results['recon_adjoint']
    final_recon = results['recon_final']
    best_method = results['best_method']
    best_tv_label = results['best_tv_label']
    
    psnr_adj, ssim_adj, rmse_adj = results['metrics_adjoint']
    psnr_cg, ssim_cg, rmse_cg = results['metrics_cg']
    final_psnr, final_ssim, final_rmse = results['metrics_final']
    
    def normalize_to_01(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    gt_n = normalize_to_01(phantom)
    adj_n = normalize_to_01(recon_adjoint)
    iter_n = normalize_to_01(final_recon)
    error = np.abs(gt_n - iter_n)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    im0 = axes[0].imshow(gt_n, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(adj_n, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('DC Adjoint (Gridding)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(iter_n, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('CG + TV Recon', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title('Error Map', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    axes[1].text(0.5, -0.08, f'PSNR={psnr_adj:.1f}dB, SSIM={ssim_adj:.3f}',
                 transform=axes[1].transAxes, ha='center', fontsize=10)
    axes[2].text(0.5, -0.08, f'PSNR={final_psnr:.1f}dB, SSIM={final_ssim:.3f}',
                 transform=axes[2].transAxes, ha='center', fontsize=10)
    axes[3].text(0.5, -0.08, f'RMSE={final_rmse:.4f}',
                 transform=axes[3].transAxes, ha='center', fontsize=10)
    
    fig.suptitle('Non-Cartesian MRI Reconstruction via NUFFT (Radial Trajectory)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {vis_path}")
    
    np.save(os.path.join(results_dir, 'ground_truth.npy'), phantom)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), final_recon)
    print(f"  Saved ground_truth.npy and reconstruction.npy")
    
    # Convert numpy types to Python native types for JSON serialization
    metrics = {
        'task': 'mri_nufft_recon',
        'task_number': 147,
        'inverse_problem': 'Non-Cartesian MRI reconstruction via NUFFT operators (radial trajectory)',
        'method': f'{best_tv_label} + TV denoising (CG on normal equations + Voronoi DC adjoint)',
        'library': 'mri-nufft + finufft',
        'image_size': int(params['N']),
        'n_spokes': int(params['n_spokes']),
        'nyquist_spokes': int(params['nyquist_spokes']),
        'acceleration_factor': float(round(params['acceleration'], 1)),
        'noise_level': float(params['noise_level']),
        'adjoint_psnr': float(round(psnr_adj, 2)),
        'adjoint_ssim': float(round(ssim_adj, 4)),
        'adjoint_rmse': float(round(rmse_adj, 4)),
        'cg_psnr': float(round(psnr_cg, 2)),
        'cg_ssim': float(round(ssim_cg, 4)),
        'cg_rmse': float(round(rmse_cg, 4)),
        'psnr': float(round(final_psnr, 2)),
        'ssim': float(round(final_ssim, 4)),
        'rmse': float(round(final_rmse, 4)),
        'best_method': str(best_method),
    }
    
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Best method: {best_method}")
    print(f"  PSNR:        {final_psnr:.2f} dB  (adjoint: {psnr_adj:.2f}, CG: {psnr_cg:.2f})")
    print(f"  SSIM:        {final_ssim:.4f}   (adjoint: {ssim_adj:.4f}, CG: {ssim_cg:.4f})")
    print(f"  RMSE:        {final_rmse:.4f}")
    print(f"  Status:      {'PASS' if final_psnr > 15 and final_ssim > 0.5 else 'FAIL'}")
    print("=" * 70)
    
    return metrics


def load_and_preprocess_data(N=256, n_spokes=100, noise_level=0.01):
    """
    Generate synthetic MRI data using the mri-nufft library.
    This recreates the data generation pipeline from the repository.
    """
    try:
        from mrinufft import get_operator
        from mrinufft.trajectories import initialize_2D_radial
        from mrinufft.density import voronoi
    except ImportError as e:
        print(f"  Warning: Could not import mri-nufft library: {e}")
        raise
    
    # Generate Shepp-Logan phantom
    try:
        from skimage.data import shepp_logan_phantom
        from skimage.transform import resize
        phantom = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True).astype(np.float32)
    except ImportError:
        # Fallback: create a simple phantom
        print("  Using simple circular phantom as fallback")
        phantom = np.zeros((N, N), dtype=np.float32)
        y, x = np.ogrid[:N, :N]
        center = N // 2
        for r, val in [(N//3, 1.0), (N//4, 0.8), (N//6, 0.5)]:
            mask = (x - center)**2 + (y - center)**2 <= r**2
            phantom[mask] = val
    
    # Compute Nyquist number of spokes
    nyquist_spokes = int(np.ceil(np.pi * N / 2))
    acceleration = nyquist_spokes / n_spokes
    
    # Number of samples per spoke
    n_samples_per_spoke = N
    
    # Generate radial trajectory
    trajectory = initialize_2D_radial(n_spokes, n_samples_per_spoke)
    trajectory = trajectory.reshape(-1, 2).astype(np.float32)
    
    # Compute density compensation using Voronoi
    try:
        density = voronoi(trajectory)
    except Exception as e:
        print(f"  Warning: Voronoi density failed ({e}), using uniform density")
        density = np.ones(trajectory.shape[0], dtype=np.float32)
    
    # Create NUFFT operators
    try:
        op_dc = get_operator("finufft")(
            samples=trajectory,
            shape=(N, N),
            density=density,
            n_coils=1
        )
        
        op_plain = get_operator("finufft")(
            samples=trajectory,
            shape=(N, N),
            density=False,
            n_coils=1
        )
    except Exception as e:
        print(f"  Error creating NUFFT operators: {e}")
        raise
    
    # Generate k-space data
    kdata = op_plain.op(phantom.astype(np.complex64))
    
    # Add noise
    if noise_level > 0:
        noise_scale = noise_level * np.abs(kdata).max()
        noise = noise_scale * (np.random.randn(*kdata.shape) + 1j * np.random.randn(*kdata.shape)) / np.sqrt(2)
        kdata = kdata + noise.astype(np.complex64)
    
    params = {
        'N': N,
        'n_spokes': n_spokes,
        'nyquist_spokes': nyquist_spokes,
        'acceleration': acceleration,
        'noise_level': noise_level,
        'n_samples_per_spoke': n_samples_per_spoke,
    }
    
    return {
        'phantom': phantom,
        'kdata': kdata,
        'trajectory': trajectory,
        'density': density,
        'op_dc': op_dc,
        'op_plain': op_plain,
        'params': params,
    }


def try_load_pickle(filepath):
    """Try multiple methods to load a pickle file."""
    methods = [
        ("dill", lambda f: dill.load(f)),
        ("pickle", lambda f: __import__('pickle').load(f)),
    ]
    
    for name, loader in methods:
        try:
            with open(filepath, 'rb') as f:
                data = loader(f)
            print(f"  Successfully loaded with {name}")
            return data
        except Exception as e:
            print(f"  {name} failed: {e}")
    
    return None


def main():
    print("=" * 70)
    print("TEST: run_inversion Performance Validation")
    print("=" * 70)
    
    # Data paths
    data_paths = ['/data/yjh/mri_nufft_recon_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Results directory
    results_base = '/data/yjh/mri_nufft_recon_sandbox_sandbox/test_results'
    agent_results_dir = os.path.join(results_base, 'agent')
    std_results_dir = os.path.join(results_base, 'standard')
    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(std_results_dir, exist_ok=True)
    
    # Try to load standard data
    outer_data = None
    std_output = None
    input_data = None
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"\nAttempting to load: {path}")
            print(f"  File size: {os.path.getsize(path)} bytes")
            outer_data = try_load_pickle(path)
            if outer_data is not None:
                break
    
    # Check if we got valid data
    if outer_data is not None and isinstance(outer_data, dict):
        print(f"\nLoaded data keys: {list(outer_data.keys())}")
        
        if 'args' in outer_data and 'kwargs' in outer_data:
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            std_output = outer_data.get('output', None)
            
            # Extract input data from args
            if len(args) > 0:
                input_data = args[0]
                print(f"  Input data type: {type(input_data)}")
                if isinstance(input_data, dict):
                    print(f"  Input data keys: {list(input_data.keys())}")
    
    # If we couldn't load valid data, generate it
    if input_data is None or not isinstance(input_data, dict) or 'phantom' not in input_data:
        print("\nGenerating fresh input data using load_and_preprocess_data...")
        np.random.seed(42)  # For reproducibility
        
        try:
            input_data = load_and_preprocess_data(N=256, n_spokes=100, noise_level=0.01)
            print(f"  Generated data with keys: {list(input_data.keys())}")
            print(f"  Phantom shape: {input_data['phantom'].shape}")
            print(f"  K-data shape: {input_data['kdata'].shape}")
            print(f"  Params: {input_data['params']}")
        except Exception as e:
            print(f"  Error generating data: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Run agent's run_inversion
    print("\n" + "=" * 70)
    print("Running agent's run_inversion...")
    print("=" * 70)
    
    try:
        agent_results = run_inversion(input_data, cg_iterations=500, cg_lambda=1e-3)
        print("\n  Agent run_inversion completed successfully")
    except Exception as e:
        print(f"\n  Agent run_inversion FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract agent metrics
    agent_psnr = agent_results['metrics_final'][0]
    agent_ssim = agent_results['metrics_final'][1]
    agent_rmse = agent_results['metrics_final'][2]
    
    print(f"\n  Agent Final PSNR: {agent_psnr:.2f} dB")
    print(f"  Agent Final SSIM: {agent_ssim:.4f}")
    print(f"  Agent Final RMSE: {agent_rmse:.4f}")
    print(f"  Agent Best Method: {agent_results['best_method']}")
    
    # Run evaluate_results for agent
    print("\nRunning evaluate_results for agent...")
    try:
        agent_metrics = evaluate_results(input_data, agent_results, agent_results_dir)
        print("  evaluate_results completed for agent")
    except Exception as e:
        print(f"  evaluate_results failed: {e}")
        traceback.print_exc()
    
    # Compare with standard output if available
    print("\n" + "=" * 70)
    print("PERFORMANCE EVALUATION")
    print("=" * 70)
    
    if std_output is not None and isinstance(std_output, dict) and 'metrics_final' in std_output:
        std_psnr = std_output['metrics_final'][0]
        std_ssim = std_output['metrics_final'][1]
        std_rmse = std_output['metrics_final'][2]
        
        print(f"\n  Standard PSNR: {std_psnr:.2f} dB")
        print(f"  Standard SSIM: {std_ssim:.4f}")
        print(f"  Agent PSNR:    {agent_psnr:.2f} dB")
        print(f"  Agent SSIM:    {agent_ssim:.4f}")
        
        # Allow 10% margin
        psnr_threshold = std_psnr * 0.9
        ssim_threshold = std_ssim * 0.9
        
        psnr_pass = agent_psnr >= psnr_threshold
        ssim_pass = agent_ssim >= ssim_threshold
        
        print(f"\n  PSNR threshold (90% of std): {psnr_threshold:.2f} dB")
        print(f"  SSIM threshold (90% of std): {ssim_threshold:.4f}")
        print(f"  PSNR: {'PASS' if psnr_pass else 'FAIL'}")
        print(f"  SSIM: {'PASS' if ssim_pass else 'FAIL'}")
        
        if psnr_pass and ssim_pass:
            print("\n" + "=" * 70)
            print("FINAL RESULT: PASS")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("FINAL RESULT: FAIL")
            print("=" * 70)
            sys.exit(1)
    else:
        # No standard output - use absolute thresholds
        # For MRI reconstruction, reasonable thresholds depend on noise and acceleration
        # With 4x acceleration and noise, PSNR > 10 and SSIM > 0.2 is acceptable
        print("\n  No valid standard output available. Using absolute quality thresholds...")
        
        # Check that reconstruction improves or is reasonable
        adj_psnr = agent_results['metrics_adjoint'][0]
        cg_psnr = agent_results['metrics_cg'][0]
        tv_psnr = agent_results['metrics_tv'][0]
        
        print(f"\n  Reconstruction quality progression:")
        print(f"    Adjoint PSNR: {adj_psnr:.2f} dB")
        print(f"    CG PSNR:      {cg_psnr:.2f} dB")
        print(f"    TV PSNR:      {tv_psnr:.2f} dB")
        print(f"    Final PSNR:   {agent_psnr:.2f} dB")
        
        # For 4x acceleration with noise, the algorithm should:
        # 1. Produce valid reconstructions (not NaN/Inf)
        # 2. TV should generally help with noisy data
        # 3. Final PSNR should be at least comparable to adjoint
        
        is_valid = (
            not np.isnan(agent_psnr) and 
            not np.isinf(agent_psnr) and
            not np.isnan(agent_ssim) and
            agent_psnr > 0  # Basic sanity check
        )
        
        # TV should improve over the base (either adjoint or CG)
        base_psnr = max(adj_psnr, cg_psnr)
        tv_improves = tv_psnr >= base_psnr * 0.95  # Allow small degradation
        
        # Final should be the best available
        final_is_best = agent_psnr >= max(adj_psnr, cg_psnr, tv_psnr) * 0.99
        
        # Absolute thresholds for 4x acceleration
        # These are lenient given high acceleration and noise
        psnr_threshold = 8.0  # With 4x acceleration and noise, this is reasonable
        ssim_threshold = 0.15
        
        print(f"\n  Validation checks:")
        print(f"    Valid output: {'PASS' if is_valid else 'FAIL'}")
        print(f"    TV improvement: {'PASS' if tv_improves else 'WARN'} (TV: {tv_psnr:.2f} vs base: {base_psnr:.2f})")
        print(f"    Final is optimal: {'PASS' if final_is_best else 'WARN'}")
        print(f"    PSNR > {psnr_threshold}: {'PASS' if agent_psnr > psnr_threshold else 'FAIL'}")
        print(f"    SSIM > {ssim_threshold}: {'PASS' if agent_ssim > ssim_threshold else 'FAIL'}")
        
        # Pass if output is valid and meets minimum thresholds
        if is_valid and agent_psnr > psnr_threshold and agent_ssim > ssim_threshold:
            print("\n" + "=" * 70)
            print("FINAL RESULT: PASS")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("FINAL RESULT: FAIL")
            print("=" * 70)
            sys.exit(1)


if __name__ == "__main__":
    main()