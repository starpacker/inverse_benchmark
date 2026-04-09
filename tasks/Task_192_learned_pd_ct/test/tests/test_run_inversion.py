import sys
import os
import dill
import numpy as np
import traceback

# Import the target function from agent module
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# Inject the referee evaluation function
def evaluate_results(ground_truth, reconstruction, sinogram_noisy, n_angles, results_dir=None):
    """
    Evaluate reconstruction results and save outputs.
    
    Parameters:
    -----------
    ground_truth : ndarray
        Ground truth image
    reconstruction : ndarray
        Reconstructed image
    sinogram_noisy : ndarray
        Noisy sinogram input
    n_angles : int
        Number of projection angles (for plot title)
    results_dir : str or None
        Directory to save results. If None, uses ./results
    
    Returns:
    --------
    dict containing:
        - 'psnr_db': Peak Signal-to-Noise Ratio in dB
        - 'ssim': Structural Similarity Index
        - 'cc': Correlation Coefficient
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    gt32 = ground_truth.astype('float32')
    recon = reconstruction.astype('float32')
    
    # Compute metrics
    data_range = float(gt32.max() - gt32.min())
    psnr_val = float(psnr_fn(gt32, recon, data_range=data_range))
    ssim_val = float(ssim_fn(gt32, recon, data_range=data_range))
    cc_val = float(np.corrcoef(gt32.ravel(), recon.ravel())[0, 1])
    
    print(f"\n{'='*50}")
    print(f"  PSNR  = {psnr_val:.2f} dB")
    print(f"  SSIM  = {ssim_val:.4f}")
    print(f"  CC    = {cc_val:.4f}")
    print(f"{'='*50}\n")
    
    # Save numerical results
    metrics = {
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "cc": round(cc_val, 4),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt32)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon)
    np.save(os.path.join(results_dir, "input.npy"), sinogram_noisy.astype('float32'))
    print("[INFO] Saved metrics.json, ground_truth.npy, reconstruction.npy, input.npy")
    
    # Visualization (2×2 panel)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Ground truth
    im0 = axes[0, 0].imshow(gt32, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Ground Truth Phantom", fontsize=13)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # (b) Noisy sinogram
    im1 = axes[0, 1].imshow(sinogram_noisy, cmap='gray', aspect='auto')
    axes[0, 1].set_title("(b) Noisy Sinogram (Input)", fontsize=13)
    axes[0, 1].set_xlabel("Detector pixel")
    axes[0, 1].set_ylabel("Projection angle index")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # (c) TV-PDHG reconstruction
    im2 = axes[1, 0].imshow(recon, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"(c) FBP + TV-PDHG Reconstruction\nPSNR={psnr_val:.1f} dB, SSIM={ssim_val:.3f}",
        fontsize=13,
    )
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # (d) Error map
    error_map = np.abs(gt32 - recon)
    im3 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0,
                             vmax=max(error_map.max(), 0.01))
    axes[1, 1].set_title("(d) Absolute Error |GT − Recon|", fontsize=13)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    fig.suptitle(
        "Task 192: CT Reconstruction via FBP + TV-PDHG\n"
        f"256×256 Shepp-Logan, {n_angles} angles, 1% noise",
        fontsize=15,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved figure → {fig_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/learned_pd_ct_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"[INFO] Outer data files: {outer_data_files}")
    print(f"[INFO] Inner data files: {inner_data_files}")
    
    try:
        # Load the primary (outer) data
        if not outer_data_files:
            print("[ERROR] No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"[INFO] Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Running agent's run_inversion with args shape info...")
        
        # Execute the agent's function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if there are inner data files (chained execution)
        if inner_data_files:
            # Chained execution pattern
            print("[INFO] Detected chained execution pattern")
            inner_data_path = inner_data_files[0]
            print(f"[INFO] Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator/function
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                print("[WARNING] Agent output is not callable, using direct output")
                final_result = agent_output
        else:
            # Direct execution pattern
            print("[INFO] Direct execution pattern")
            final_result = agent_output
            std_result = std_output
        
        print(f"[INFO] Agent result type: {type(final_result)}")
        print(f"[INFO] Standard result type: {type(std_result)}")
        
        # Extract reconstruction from results (they are dicts)
        if isinstance(final_result, dict) and 'reconstruction' in final_result:
            agent_reconstruction = final_result['reconstruction']
        else:
            agent_reconstruction = final_result
            
        if isinstance(std_result, dict) and 'reconstruction' in std_result:
            std_reconstruction = std_result['reconstruction']
        else:
            std_reconstruction = std_result
        
        # For evaluation, we need ground truth, sinogram_noisy, and n_angles
        # Extract sinogram_noisy from input args
        sinogram_noisy = args[0] if len(args) > 0 else kwargs.get('sinogram_noisy')
        theta_angles = args[1] if len(args) > 1 else kwargs.get('theta_angles')
        n_angles = len(theta_angles) if theta_angles is not None else 0
        
        # We need ground truth for evaluation
        # Since we don't have ground truth directly, we'll use the standard reconstruction as reference
        # and compare metrics between agent and standard outputs
        
        # Create results directories
        agent_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_agent")
        std_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_std")
        
        # Since we don't have the actual ground truth, we'll compute comparative metrics
        # Use std_reconstruction as reference (pseudo ground truth)
        print("\n[INFO] Evaluating agent reconstruction against standard...")
        
        # Compute direct comparison metrics
        agent_recon = agent_reconstruction.astype('float32')
        std_recon = std_reconstruction.astype('float32')
        
        # Data range for metrics
        data_range = float(max(std_recon.max() - std_recon.min(), 1e-6))
        
        # Compute PSNR and SSIM between agent and standard
        psnr_comparison = float(psnr_fn(std_recon, agent_recon, data_range=data_range))
        ssim_comparison = float(ssim_fn(std_recon, agent_recon, data_range=data_range))
        cc_comparison = float(np.corrcoef(std_recon.ravel(), agent_recon.ravel())[0, 1])
        
        print(f"\n{'='*60}")
        print("COMPARISON METRICS (Agent vs Standard):")
        print(f"  PSNR  = {psnr_comparison:.2f} dB")
        print(f"  SSIM  = {ssim_comparison:.4f}")
        print(f"  CC    = {cc_comparison:.4f}")
        print(f"{'='*60}\n")
        
        # Also compute self-consistency metrics
        # MSE between agent and standard
        mse = np.mean((agent_recon - std_recon) ** 2)
        max_diff = np.max(np.abs(agent_recon - std_recon))
        
        print(f"[INFO] MSE between agent and standard: {mse:.6f}")
        print(f"[INFO] Max absolute difference: {max_diff:.6f}")
        
        # Verify reconstruction quality
        # For a valid implementation, PSNR should be very high (ideally infinite for identical outputs)
        # SSIM should be close to 1.0
        # We use reasonable thresholds for numerical implementations
        
        PSNR_THRESHOLD = 30.0  # dB - very high similarity expected
        SSIM_THRESHOLD = 0.95   # Close to 1.0
        CC_THRESHOLD = 0.95     # High correlation expected
        
        success = True
        
        if psnr_comparison < PSNR_THRESHOLD:
            print(f"[WARNING] PSNR {psnr_comparison:.2f} dB is below threshold {PSNR_THRESHOLD} dB")
            # Allow some tolerance for numerical differences
            if psnr_comparison < PSNR_THRESHOLD * 0.8:
                success = False
        
        if ssim_comparison < SSIM_THRESHOLD:
            print(f"[WARNING] SSIM {ssim_comparison:.4f} is below threshold {SSIM_THRESHOLD}")
            if ssim_comparison < SSIM_THRESHOLD * 0.9:
                success = False
        
        if cc_comparison < CC_THRESHOLD:
            print(f"[WARNING] CC {cc_comparison:.4f} is below threshold {CC_THRESHOLD}")
            if cc_comparison < CC_THRESHOLD * 0.9:
                success = False
        
        # Additional check: verify output structure is correct
        if isinstance(final_result, dict):
            required_keys = ['reconstruction', 'fbp_reconstruction']
            for key in required_keys:
                if key not in final_result:
                    print(f"[ERROR] Missing key '{key}' in agent output")
                    success = False
        else:
            print(f"[ERROR] Agent output should be a dict, got {type(final_result)}")
            success = False
        
        # Check reconstruction value range
        if agent_reconstruction.min() < -0.1 or agent_reconstruction.max() > 1.1:
            print(f"[WARNING] Reconstruction values outside expected [0,1] range: [{agent_reconstruction.min():.4f}, {agent_reconstruction.max():.4f}]")
        
        print(f"\n{'='*60}")
        print(f"Scores -> Agent PSNR: {psnr_comparison:.2f}, SSIM: {ssim_comparison:.4f}, CC: {cc_comparison:.4f}")
        print(f"Thresholds -> PSNR: {PSNR_THRESHOLD}, SSIM: {SSIM_THRESHOLD}, CC: {CC_THRESHOLD}")
        print(f"{'='*60}\n")
        
        if success:
            print("[SUCCESS] Agent implementation matches standard with acceptable tolerance")
            sys.exit(0)
        else:
            print("[FAILURE] Agent implementation deviates significantly from standard")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Exception during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()