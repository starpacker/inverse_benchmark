import os

import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
