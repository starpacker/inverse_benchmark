import sys
import os
import dill
import numpy as np
import traceback
import json

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Inject the referee (evaluation logic)
def normalize(x):
    """Normalize array to [0, 1] range."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-15)

def evaluate_results(data_dict, result_dict):
    """
    Evaluate the inversion results and save outputs.
    
    Computes quality metrics (PSNR, SSIM, correlation coefficient, relative errors)
    and saves results to disk including metrics JSON, numpy arrays, and visualization.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing preprocessed data including ground truth
    result_dict : dict
        Dictionary containing inversion results
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all computed metrics
    """
    print("[5/5] Evaluating ...")
    
    gt_img = data_dict['gt_img']
    noisy_img = data_dict['noisy_img']
    V_pot = data_dict['V_pot']
    true_df = data_dict['true_defocus']
    true_t = data_dict['true_thickness']
    
    recon_img = result_dict['recon_img']
    est_df = result_dict['estimated_defocus']
    est_t = result_dict['estimated_thickness']
    
    # Normalize images for metric computation
    gt_n = normalize(gt_img)
    recon_n = normalize(recon_img)
    noisy_n = normalize(noisy_img)

    # Compute quality metrics for reconstruction
    psnr_r = peak_signal_noise_ratio(gt_n, recon_n, data_range=1.0)
    ssim_r = structural_similarity(gt_n, recon_n, data_range=1.0)
    cc_r = float(np.corrcoef(gt_n.ravel(), recon_n.ravel())[0, 1])

    # Compute quality metrics for noisy observation
    psnr_n = peak_signal_noise_ratio(gt_n, noisy_n, data_range=1.0)
    ssim_n = structural_similarity(gt_n, noisy_n, data_range=1.0)

    # Compute relative errors for parameter estimation
    re_df = abs(est_df - true_df) / abs(true_df)
    re_t = abs(est_t - true_t) / abs(true_t)

    # Print results
    print(f"\n{'─'*55}")
    print(f"  TRUE : df={true_df:.1f} nm, t={true_t:.2f} nm")
    print(f"  EST  : df={est_df:.3f} nm, t={est_t:.4f} nm")
    print(f"  RE   : df={re_df:.6f}, t={re_t:.6f}")
    print(f"  Recon: PSNR={psnr_r:.2f}, SSIM={ssim_r:.4f}, CC={cc_r:.4f}")
    print(f"  Noisy: PSNR={psnr_n:.2f}, SSIM={ssim_n:.4f}")
    print(f"{'─'*55}\n")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Prepare metrics dictionary
    metrics = {
        "task_id": 178,
        "task_name": "abtem_sim",
        "inverse_problem": "HRTEM inverse parameter estimation (defocus + thickness)",
        "true_defocus_nm": true_df,
        "true_thickness_nm": true_t,
        "estimated_defocus_nm": round(float(est_df), 3),
        "estimated_thickness_nm": round(float(est_t), 4),
        "defocus_relative_error": round(float(re_df), 6),
        "thickness_relative_error": round(float(re_t), 6),
        "reconstruction_PSNR_dB": round(float(psnr_r), 2),
        "reconstruction_SSIM": round(float(ssim_r), 4),
        "reconstruction_CC": round(float(cc_r), 4),
        "noisy_PSNR_dB": round(float(psnr_n), 2),
        "noisy_SSIM": round(float(ssim_n), 4),
    }
    
    # Save metrics JSON
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/metrics.json")

    # Save numpy arrays
    np.save("results/ground_truth.npy", gt_img)
    np.save("results/noisy_observation.npy", noisy_img)
    np.save("results/reconstruction.npy", recon_img)
    np.save("results/projected_potential.npy", V_pot)
    print("Saved .npy arrays")

    # Create visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    im0 = axes[0, 0].imshow(gt_n, cmap="gray", origin="lower")
    axes[0, 0].set_title("(a) GT noiseless HRTEM image", fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(noisy_n, cmap="gray", origin="lower")
    axes[0, 1].set_title("(b) Noisy observation", fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(recon_n, cmap="gray", origin="lower")
    axes[1, 0].set_title(f"(c) Best-fit image\ndf={est_df:.2f} nm, t={est_t:.3f} nm", fontsize=11)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    err = np.abs(gt_n - recon_n)
    im3 = axes[1, 1].imshow(err, cmap="hot", origin="lower")
    axes[1, 1].set_title(f"(d) |GT − Recon| error\nPSNR={psnr_r:.1f} dB, SSIM={ssim_r:.3f}", fontsize=11)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Task 178: abtem_sim — HRTEM Inverse Parameter Estimation\n"
        f"True: df={true_df} nm, t={true_t} nm  |  Est: df={est_df:.2f} nm, t={est_t:.3f} nm\n"
        f"Defocus RE={re_df:.4f}, Thickness RE={re_t:.4f}",
        fontsize=12, fontweight="bold", y=1.01)

    for ax in axes.ravel():
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig("results/reconstruction_result.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/reconstruction_result.png\n\nDone.")

    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/abtem_sim_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Parse paths to identify outer and inner data
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    # Load outer (primary) data
    if outer_data_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_data_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args, kwargs, and expected output from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Args length: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Execute the agent's run_inversion function
    try:
        print("\n=== Running agent's run_inversion ===")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent's run_inversion completed successfully.")
    except Exception as e:
        print(f"ERROR: Agent's run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution pattern)
    if inner_data_paths:
        # Chained execution pattern
        print("\n=== Chained Execution Pattern Detected ===")
        
        for inner_path in inner_data_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator
            try:
                print("\n=== Running agent's returned operator ===")
                final_result = agent_output(*inner_args, **inner_kwargs)
                print("Agent's operator completed successfully.")
            except Exception as e:
                print(f"ERROR: Agent's operator failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution pattern
        print("\n=== Direct Execution Pattern ===")
        final_result = agent_output
        std_result = std_output
    
    # For evaluation, we need the input data_dict which contains ground truth info
    # The input data_dict is the first argument to run_inversion
    if len(args) > 0:
        data_dict = args[0]
    else:
        data_dict = kwargs.get('data_dict', None)
    
    if data_dict is None:
        print("ERROR: Could not find data_dict in inputs.")
        sys.exit(1)
    
    # Evaluate agent's result
    try:
        print("\n=== Evaluating Agent's Result ===")
        metrics_agent = evaluate_results(data_dict, final_result)
        print(f"Agent metrics: PSNR={metrics_agent['reconstruction_PSNR_dB']}, SSIM={metrics_agent['reconstruction_SSIM']}")
    except Exception as e:
        print(f"ERROR: Failed to evaluate agent's result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    try:
        print("\n=== Evaluating Standard Result ===")
        metrics_std = evaluate_results(data_dict, std_result)
        print(f"Standard metrics: PSNR={metrics_std['reconstruction_PSNR_dB']}, SSIM={metrics_std['reconstruction_SSIM']}")
    except Exception as e:
        print(f"ERROR: Failed to evaluate standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare scores
    # For PSNR and SSIM, higher is better
    score_agent_psnr = metrics_agent['reconstruction_PSNR_dB']
    score_std_psnr = metrics_std['reconstruction_PSNR_dB']
    
    score_agent_ssim = metrics_agent['reconstruction_SSIM']
    score_std_ssim = metrics_std['reconstruction_SSIM']
    
    # Also compare relative errors (lower is better)
    re_df_agent = metrics_agent['defocus_relative_error']
    re_t_agent = metrics_agent['thickness_relative_error']
    re_df_std = metrics_std['defocus_relative_error']
    re_t_std = metrics_std['thickness_relative_error']
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"PSNR -> Agent: {score_agent_psnr:.2f} dB, Standard: {score_std_psnr:.2f} dB")
    print(f"SSIM -> Agent: {score_agent_ssim:.4f}, Standard: {score_std_ssim:.4f}")
    print(f"Defocus RE -> Agent: {re_df_agent:.6f}, Standard: {re_df_std:.6f}")
    print(f"Thickness RE -> Agent: {re_t_agent:.6f}, Standard: {re_t_std:.6f}")
    print(f"{'='*60}")
    
    # Determine success
    # Allow 10% margin of error for PSNR and SSIM (higher is better)
    # For relative errors, agent should not be significantly worse (allow 20% margin)
    
    psnr_threshold = score_std_psnr * 0.9  # Agent should achieve at least 90% of standard PSNR
    ssim_threshold = score_std_ssim * 0.9  # Agent should achieve at least 90% of standard SSIM
    
    success = True
    
    if score_agent_psnr < psnr_threshold:
        print(f"FAIL: Agent PSNR ({score_agent_psnr:.2f}) is below threshold ({psnr_threshold:.2f})")
        success = False
    else:
        print(f"PASS: Agent PSNR ({score_agent_psnr:.2f}) meets threshold ({psnr_threshold:.2f})")
    
    if score_agent_ssim < ssim_threshold:
        print(f"FAIL: Agent SSIM ({score_agent_ssim:.4f}) is below threshold ({ssim_threshold:.4f})")
        success = False
    else:
        print(f"PASS: Agent SSIM ({score_agent_ssim:.4f}) meets threshold ({ssim_threshold:.4f})")
    
    # Check relative errors (allow agent to be at most 2x worse than standard)
    if re_df_agent > re_df_std * 2.0 and re_df_agent > 0.1:  # Only fail if significantly worse and error > 10%
        print(f"WARNING: Agent defocus RE ({re_df_agent:.6f}) is significantly worse than standard ({re_df_std:.6f})")
    
    if re_t_agent > re_t_std * 2.0 and re_t_agent > 0.1:  # Only fail if significantly worse and error > 10%
        print(f"WARNING: Agent thickness RE ({re_t_agent:.6f}) is significantly worse than standard ({re_t_std:.6f})")
    
    if success:
        print("\n*** TEST PASSED ***")
        sys.exit(0)
    else:
        print("\n*** TEST FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()