import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Define RESULTS_DIR
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the evaluate_results function (Reference B)
def evaluate_results(phantom, recon_fbp, recon_cgls, recon_tv, sinogram_noisy,
                     n_angles, noise_level):
    """
    Evaluate reconstruction quality and save results.
    """
    
    def compute_metrics(gt, recon):
        """Compute PSNR, SSIM, RMSE between ground truth and reconstruction."""
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
        recon_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)
        
        psnr = peak_signal_noise_ratio(gt_n, recon_n, data_range=1.0)
        ssim = structural_similarity(gt_n, recon_n, data_range=1.0)
        rmse = np.sqrt(np.mean((gt_n - recon_n) ** 2))
        return psnr, ssim, rmse
    
    def norm01(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    # Compute metrics for each method
    psnr_fbp, ssim_fbp, rmse_fbp = compute_metrics(phantom, recon_fbp)
    print(f"\n  FBP  — PSNR: {psnr_fbp:.2f} dB, "
          f"SSIM: {ssim_fbp:.4f}, RMSE: {rmse_fbp:.4f}")
    
    psnr_cgls, ssim_cgls, rmse_cgls = compute_metrics(phantom, recon_cgls)
    print(f"  CGLS — PSNR: {psnr_cgls:.2f} dB, "
          f"SSIM: {ssim_cgls:.4f}, RMSE: {rmse_cgls:.4f}")
    
    psnr_tv, ssim_tv, rmse_tv = compute_metrics(phantom, recon_tv)
    print(f"  TV   — PSNR: {psnr_tv:.2f} dB, "
          f"SSIM: {ssim_tv:.4f}, RMSE: {rmse_tv:.4f}")
    
    # Determine best method
    results = {
        'FBP':      (recon_fbp,  psnr_fbp,  ssim_fbp,  rmse_fbp),
        'CGLS':     (recon_cgls, psnr_cgls, ssim_cgls, rmse_cgls),
        'TV-FISTA': (recon_tv,   psnr_tv,   ssim_tv,   rmse_tv),
    }
    best_name = max(results, key=lambda k: results[k][1])
    best_recon, best_psnr, best_ssim, best_rmse = results[best_name]
    
    print(f"\n★ Best method: {best_name}")
    print(f"  PSNR: {best_psnr:.2f} dB | "
          f"SSIM: {best_ssim:.4f} | RMSE: {best_rmse:.4f}")
    
    # Save outputs
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), phantom)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_recon)
    
    metrics = {
        "PSNR": round(float(best_psnr), 2),
        "SSIM": round(float(best_ssim), 4),
        "RMSE": round(float(best_rmse), 4),
        "best_method": best_name,
        "n_angles": n_angles,
        "noise_level": noise_level,
        "FBP": {
            "PSNR": round(float(psnr_fbp), 2),
            "SSIM": round(float(ssim_fbp), 4),
            "RMSE": round(float(rmse_fbp), 4),
        },
        "CGLS": {
            "PSNR": round(float(psnr_cgls), 2),
            "SSIM": round(float(ssim_cgls), 4),
            "RMSE": round(float(rmse_cgls), 4),
        },
        "TV-FISTA": {
            "PSNR": round(float(psnr_tv), 2),
            "SSIM": round(float(ssim_tv), 4),
            "RMSE": round(float(rmse_tv), 4),
        },
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.json")
    
    return metrics


def compute_reconstruction_metrics(phantom, recon):
    """Compute metrics between ground truth and a single reconstruction."""
    gt_n = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-12)
    recon_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)
    
    psnr = peak_signal_noise_ratio(gt_n, recon_n, data_range=1.0)
    ssim = structural_similarity(gt_n, recon_n, data_range=1.0)
    rmse = np.sqrt(np.mean((gt_n - recon_n) ** 2))
    return psnr, ssim, rmse


def main():
    # Data paths provided
    data_paths = ['/data/yjh/cil_tomo_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 60)
    print("Loading test data...")
    print("=" * 60)
    
    # Analyze files to determine execution pattern
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_data_paths.append(path)
            else:
                outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    print(f"\nLoading outer data from: {outer_data_path}")
    with open(outer_data_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'unknown')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {func_name}")
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Extract common parameters from the first call
    # The data contains: sinogram, theta, output_size, adjoint_scale, method, ...
    sinogram = args[0] if len(args) > 0 else kwargs.get('sinogram')
    theta = args[1] if len(args) > 1 else kwargs.get('theta')
    output_size = args[2] if len(args) > 2 else kwargs.get('output_size')
    adjoint_scale = args[3] if len(args) > 3 else kwargs.get('adjoint_scale')
    
    print(f"\nSinogram shape: {sinogram.shape}")
    print(f"Theta shape: {theta.shape}")
    print(f"Output size: {output_size}")
    print(f"Adjoint scale: {adjoint_scale}")
    
    # We need to generate the phantom for evaluation
    # Create Shepp-Logan phantom for comparison
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    
    phantom_full = shepp_logan_phantom()
    phantom = resize(phantom_full, (output_size, output_size), anti_aliasing=True)
    
    n_angles = len(theta)
    # Estimate noise level from sinogram
    noise_level = 0.02  # Default assumption
    
    all_passed = True
    
    try:
        # Run all three methods with agent's implementation
        print("\n" + "=" * 60)
        print("Running agent's run_inversion with FBP method...")
        print("=" * 60)
        recon_fbp_agent = run_inversion(sinogram, theta, output_size, adjoint_scale, 
                                         method='FBP')
        
        print("\n" + "=" * 60)
        print("Running agent's run_inversion with CGLS method...")
        print("=" * 60)
        recon_cgls_agent = run_inversion(sinogram, theta, output_size, adjoint_scale, 
                                          method='CGLS', cgls_iter=30)
        
        print("\n" + "=" * 60)
        print("Running agent's run_inversion with TV-FISTA method...")
        print("=" * 60)
        recon_tv_agent = run_inversion(sinogram, theta, output_size, adjoint_scale, 
                                        method='TV-FISTA', fista_iter=100, lam_tv=0.003)
        
        # Evaluate agent results
        print("\n" + "=" * 60)
        print("Evaluating agent's results...")
        print("=" * 60)
        
        agent_metrics = evaluate_results(phantom, recon_fbp_agent, recon_cgls_agent, 
                                          recon_tv_agent, sinogram, n_angles, noise_level)
        
        # Compute metrics for the standard output (single reconstruction)
        print("\n" + "=" * 60)
        print("Comparing with standard output...")
        print("=" * 60)
        
        if std_output is not None:
            std_psnr, std_ssim, std_rmse = compute_reconstruction_metrics(phantom, std_output)
            print(f"\nStandard output metrics:")
            print(f"  PSNR: {std_psnr:.2f} dB, SSIM: {std_ssim:.4f}, RMSE: {std_rmse:.4f}")
            
            # Get corresponding agent metric based on the method used in std_output
            # The standard data used one specific method, compare against best agent result
            agent_psnr = agent_metrics['PSNR']
            agent_ssim = agent_metrics['SSIM']
            agent_rmse = agent_metrics['RMSE']
            
            print(f"\nAgent best output metrics:")
            print(f"  PSNR: {agent_psnr:.2f} dB, SSIM: {agent_ssim:.4f}, RMSE: {agent_rmse:.4f}")
            
            # Tolerance for comparison (allow 10% degradation)
            psnr_tolerance = 0.90
            ssim_tolerance = 0.90
            rmse_tolerance = 1.10  # RMSE lower is better, so allow 10% higher
            
            print("\n" + "=" * 60)
            print("Performance Comparison (Agent vs Standard)")
            print("=" * 60)
            
            # Check PSNR
            if agent_psnr >= std_psnr * psnr_tolerance:
                print(f"✓ PASS: Agent PSNR {agent_psnr:.2f} >= {std_psnr * psnr_tolerance:.2f} (Standard * 0.9)")
            else:
                print(f"❌ FAIL: Agent PSNR {agent_psnr:.2f} < {std_psnr * psnr_tolerance:.2f} (Standard * 0.9)")
                all_passed = False
            
            # Check SSIM
            if agent_ssim >= std_ssim * ssim_tolerance:
                print(f"✓ PASS: Agent SSIM {agent_ssim:.4f} >= {std_ssim * ssim_tolerance:.4f} (Standard * 0.9)")
            else:
                print(f"❌ FAIL: Agent SSIM {agent_ssim:.4f} < {std_ssim * ssim_tolerance:.4f} (Standard * 0.9)")
                all_passed = False
            
            # Check RMSE (lower is better)
            if agent_rmse <= std_rmse * rmse_tolerance:
                print(f"✓ PASS: Agent RMSE {agent_rmse:.4f} <= {std_rmse * rmse_tolerance:.4f} (Standard * 1.1)")
            else:
                print(f"❌ FAIL: Agent RMSE {agent_rmse:.4f} > {std_rmse * rmse_tolerance:.4f} (Standard * 1.1)")
                all_passed = False
        else:
            print("\nNo standard output available for comparison.")
            print("Using absolute quality thresholds instead.")
            
            # Use reasonable absolute thresholds
            min_psnr = 20.0
            min_ssim = 0.4
            max_rmse = 0.15
            
            if agent_metrics['PSNR'] >= min_psnr:
                print(f"✓ PASS: PSNR {agent_metrics['PSNR']:.2f} >= {min_psnr:.2f}")
            else:
                print(f"❌ FAIL: PSNR {agent_metrics['PSNR']:.2f} < {min_psnr:.2f}")
                all_passed = False
            
            if agent_metrics['SSIM'] >= min_ssim:
                print(f"✓ PASS: SSIM {agent_metrics['SSIM']:.4f} >= {min_ssim:.4f}")
            else:
                print(f"❌ FAIL: SSIM {agent_metrics['SSIM']:.4f} < {min_ssim:.4f}")
                all_passed = False
            
            if agent_metrics['RMSE'] <= max_rmse:
                print(f"✓ PASS: RMSE {agent_metrics['RMSE']:.4f} <= {max_rmse:.4f}")
            else:
                print(f"❌ FAIL: RMSE {agent_metrics['RMSE']:.4f} > {max_rmse:.4f}")
                all_passed = False
        
        # Verify all methods produce valid outputs
        print("\n" + "=" * 60)
        print("Method Validation")
        print("=" * 60)
        
        for method_name, method_metrics in [('FBP', agent_metrics['FBP']), 
                                             ('CGLS', agent_metrics['CGLS']),
                                             ('TV-FISTA', agent_metrics['TV-FISTA'])]:
            if method_metrics['PSNR'] > 10 and method_metrics['SSIM'] > 0.1:
                print(f"✓ {method_name}: Valid reconstruction (PSNR={method_metrics['PSNR']:.2f}, SSIM={method_metrics['SSIM']:.4f})")
            else:
                print(f"❌ {method_name}: Poor reconstruction (PSNR={method_metrics['PSNR']:.2f}, SSIM={method_metrics['SSIM']:.4f})")
                all_passed = False
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()