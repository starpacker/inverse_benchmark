import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion


# Inject the referee evaluation function
def evaluate_results(gt_field, reconstruction, dmd_object, true_discrete_eigenvalues, 
                     svd_rank, nx, ny, noisy_field, results_dir):
    """
    Evaluate DMD reconstruction quality and save results.
    
    Computes:
    - PSNR between ground truth and reconstruction
    - Correlation coefficient
    - MSE
    - Eigenvalue relative errors
    
    Also generates visualization and saves metrics.
    
    Parameters
    ----------
    gt_field : ndarray (n_spatial, nt)
        Ground-truth snapshot matrix
    reconstruction : ndarray (n_spatial, nt)
        DMD reconstruction
    dmd_object : DMD
        Fitted DMD object
    true_discrete_eigenvalues : ndarray
        True discrete eigenvalues for comparison
    svd_rank : int
        SVD rank used
    nx, ny : int
        Spatial grid dimensions
    noisy_field : ndarray
        Noisy input data for visualization
    results_dir : str
        Directory to save results
    
    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    gt64 = gt_field.astype(np.float64)
    re64 = reconstruction.astype(np.float64)
    
    # Align time axis (reconstructed_data may differ by 1 column)
    min_t = min(gt64.shape[1], re64.shape[1])
    gt64 = gt64[:, :min_t]
    re64 = re64[:, :min_t]
    
    # Compute MSE
    mse = np.mean((gt64 - re64) ** 2)
    
    # Compute PSNR
    data_range = gt64.max() - gt64.min()
    if mse > 0:
        psnr = 10 * np.log10(data_range ** 2 / mse)
    else:
        psnr = float("inf")
    
    # Compute correlation coefficient
    cc = float(np.corrcoef(gt64.ravel(), re64.ravel())[0, 1])
    
    # Compute eigenvalue errors
    recovered = dmd_object.eigs
    gt_all = np.concatenate([true_discrete_eigenvalues, true_discrete_eigenvalues.conj()])
    eigenvalue_errors = []
    for gt_e in gt_all:
        dists = np.abs(recovered - gt_e)
        idx = np.argmin(dists)
        rel_err = float(np.abs(recovered[idx] - gt_e) / np.abs(gt_e))
        eigenvalue_errors.append(round(rel_err, 8))
    
    metrics = {
        "psnr_db": round(float(psnr), 4),
        "correlation_coefficient": round(cc, 6),
        "mse": float(f"{mse:.10e}"),
        "eigenvalue_relative_errors": eigenvalue_errors,
        "n_modes": int(len(dmd_object.eigs)),
        "svd_rank": svd_rank,
    }
    
    print(f"\n[METRICS] PSNR  = {metrics['psnr_db']:.2f} dB")
    print(f"[METRICS] CC    = {metrics['correlation_coefficient']:.6f}")
    print(f"[METRICS] MSE   = {metrics['mse']:.2e}")
    print(f"[METRICS] Eig. rel. err = {eigenvalue_errors}")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[INFO] Saved metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_field)
    np.save(os.path.join(results_dir, "reconstruction.npy"), reconstruction)
    print("[INFO] Saved ground_truth.npy and reconstruction.npy")
    
    # Visualization
    gt_img = gt_field[:, 0].reshape(nx, ny)
    noisy_img = noisy_field[:, 0].reshape(nx, ny)
    recon_img = reconstruction[:, 0].reshape(nx, ny)
    err_img = np.abs(gt_img - recon_img)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    
    titles = [
        "(a) Ground Truth (t=0)",
        "(b) Noisy Input (t=0)",
        "(c) DMD Reconstruction (t=0)",
        "(d) Error |GT − Recon|",
    ]
    images = [gt_img, noisy_img, recon_img, err_img]
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "hot"]
    
    vmin = min(gt_img.min(), noisy_img.min(), recon_img.min())
    vmax = max(gt_img.max(), noisy_img.max(), recon_img.max())
    
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        if cmap == "hot":
            im = ax.imshow(img.T, origin="lower", cmap=cmap, aspect="equal")
        else:
            im = ax.imshow(
                img.T, origin="lower", cmap=cmap, aspect="equal",
                vmin=vmin, vmax=vmax,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(
        "Task 174 — Dynamic Mode Decomposition (PyDMD)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved visualization → {vis_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/pydmd_dmd_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    # Load outer data
    if not outer_data_files:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    outer_data_path = outer_data_files[0]
    print(f"[INFO] Loading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs and expected output
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent function
    print("[INFO] Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if we have chained execution
    if inner_data_files:
        # Chained execution pattern
        print("[INFO] Chained execution detected - loading inner data...")
        inner_data_path = inner_data_files[0]
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        if callable(agent_output):
            print("[INFO] Executing returned operator with inner args...")
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            final_result = agent_output
    else:
        # Direct execution pattern
        final_result = agent_output
        std_result = std_output
    
    print("[INFO] Execution completed successfully.")
    
    # Now we need to evaluate results
    # The evaluate_results function requires additional parameters that we need to obtain
    # from the data generation context. We need: gt_field, true_discrete_eigenvalues, nx, ny
    
    # Extract from the agent result
    agent_reconstruction = final_result.get('reconstruction', None)
    agent_dmd_object = final_result.get('dmd_object', None)
    
    std_reconstruction = std_result.get('reconstruction', None) if std_result else None
    std_dmd_object = std_result.get('dmd_object', None) if std_result else None
    
    # Get svd_rank from kwargs or args
    if 'svd_rank' in kwargs:
        svd_rank = kwargs['svd_rank']
    elif len(args) >= 2:
        svd_rank = args[1]
    else:
        svd_rank = 4  # default
    
    # Get noisy_field from args
    if len(args) >= 1:
        noisy_field = args[0]
    elif 'noisy_field' in kwargs:
        noisy_field = kwargs['noisy_field']
    else:
        print("[ERROR] Could not extract noisy_field from inputs!")
        sys.exit(1)
    
    print(f"[INFO] noisy_field shape: {noisy_field.shape}")
    print(f"[INFO] svd_rank: {svd_rank}")
    
    # Since we don't have gt_field and true_discrete_eigenvalues from the pickle,
    # we'll compare the results directly between agent and standard outputs
    
    # Comparison approach: Compare reconstructions and eigenvalues
    print("\n" + "="*60)
    print("COMPARISON: Agent vs Standard Output")
    print("="*60)
    
    if agent_reconstruction is not None and std_reconstruction is not None:
        # Align shapes
        min_t = min(agent_reconstruction.shape[1], std_reconstruction.shape[1])
        agent_recon_aligned = agent_reconstruction[:, :min_t]
        std_recon_aligned = std_reconstruction[:, :min_t]
        
        # Compute MSE between agent and standard
        mse_diff = np.mean((agent_recon_aligned - std_recon_aligned) ** 2)
        
        # Compute correlation
        cc = np.corrcoef(agent_recon_aligned.ravel(), std_recon_aligned.ravel())[0, 1]
        
        # Compute relative difference
        std_norm = np.linalg.norm(std_recon_aligned)
        if std_norm > 0:
            rel_diff = np.linalg.norm(agent_recon_aligned - std_recon_aligned) / std_norm
        else:
            rel_diff = 0.0
        
        print(f"[COMPARE] Reconstruction MSE (agent vs std): {mse_diff:.2e}")
        print(f"[COMPARE] Reconstruction Correlation: {cc:.6f}")
        print(f"[COMPARE] Reconstruction Relative Diff: {rel_diff:.6f}")
        
        # Also compare eigenvalues
        if agent_dmd_object is not None and std_dmd_object is not None:
            agent_eigs = agent_dmd_object.eigs
            std_eigs = std_dmd_object.eigs
            
            print(f"[COMPARE] Agent eigenvalues count: {len(agent_eigs)}")
            print(f"[COMPARE] Standard eigenvalues count: {len(std_eigs)}")
            
            # Compute eigenvalue matching error
            if len(agent_eigs) == len(std_eigs):
                # Sort by magnitude for comparison
                agent_eigs_sorted = np.sort(agent_eigs)
                std_eigs_sorted = np.sort(std_eigs)
                eig_diff = np.mean(np.abs(agent_eigs_sorted - std_eigs_sorted))
                print(f"[COMPARE] Mean eigenvalue difference: {eig_diff:.6e}")
        
        # Determine pass/fail
        # For reconstruction, we expect very high correlation (>0.99) and low relative diff (<0.05)
        # Since this is deterministic DMD, results should be nearly identical
        
        tolerance_cc = 0.99  # Correlation should be > 0.99
        tolerance_rel = 0.05  # Relative difference should be < 5%
        
        passed = True
        
        if cc < tolerance_cc:
            print(f"[WARN] Correlation {cc:.6f} < {tolerance_cc} threshold")
            passed = False
        
        if rel_diff > tolerance_rel:
            print(f"[WARN] Relative difference {rel_diff:.6f} > {tolerance_rel} threshold")
            passed = False
        
        print("\n" + "="*60)
        if passed:
            print("[RESULT] TEST PASSED - Agent output matches standard within tolerance")
            print("="*60)
            sys.exit(0)
        else:
            print("[RESULT] TEST FAILED - Agent output differs significantly from standard")
            print("="*60)
            sys.exit(1)
    else:
        # Cannot compare reconstructions, fall back to basic checks
        print("[WARN] Cannot compare reconstructions directly")
        
        # Check that agent output has expected keys
        expected_keys = ['dmd_object', 'reconstruction', 'modes', 'eigenvalues', 'dynamics']
        missing_keys = [k for k in expected_keys if k not in final_result]
        
        if missing_keys:
            print(f"[ERROR] Missing keys in agent output: {missing_keys}")
            sys.exit(1)
        
        # Check shapes are reasonable
        if agent_reconstruction is not None:
            print(f"[INFO] Agent reconstruction shape: {agent_reconstruction.shape}")
            if agent_reconstruction.shape[0] != noisy_field.shape[0]:
                print("[ERROR] Reconstruction spatial dimension mismatch!")
                sys.exit(1)
        
        print("[RESULT] Basic structure checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()