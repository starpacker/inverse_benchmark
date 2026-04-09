import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from itertools import permutations

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def align_endmembers(E_gt, E_rec, A_gt, A_rec):
    """
    Find optimal permutation to align estimated endmembers with GT.
    """
    R = E_gt.shape[1]

    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(R)):
        score = 0
        for i, j in enumerate(perm):
            cos_val = np.dot(E_gt[:, i], E_rec[:, j]) / (
                np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_rec[:, j]) + 1e-12
            )
            score += cos_val
        if score > best_score:
            best_score = score
            best_perm = perm

    perm_list = list(best_perm)
    E_aligned = E_rec[:, perm_list]
    A_aligned = A_rec[perm_list, :]
    return E_aligned, A_aligned, perm_list

def forward_operator(E, A):
    """
    Linear spectral mixing model: Y = E @ A
    
    Args:
        E: Endmember matrix (L x R) - spectral signatures
        A: Abundance matrix (R x P) - fractional abundances
        
    Returns:
        Y_pred: Predicted mixed spectra (L x P)
    """
    Y_pred = E @ A
    return Y_pred

def evaluate_results(data, inversion_result):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        data: Dictionary from load_and_preprocess_data
        inversion_result: Dictionary from run_inversion
        
    Returns:
        dict: Final metrics
    """
    E_gt = data['E_gt']
    A_gt = data['A_gt']
    wavelengths = data['wavelengths']
    img_size = data['img_size']
    
    E_rec = inversion_result['E_rec']
    A_rec = inversion_result['A_rec']
    metrics = inversion_result['metrics']
    method = inversion_result['method']
    
    # Forward model verification
    print("\n[STAGE 2] Forward — Linear Mixing Model Y = E·A + N")
    Y_verify = forward_operator(E_gt, A_gt)
    Y_clean = data['Y_clean']
    fwd_error = np.linalg.norm(Y_clean - Y_verify) / np.linalg.norm(Y_clean)
    print(f"  Forward model verification error: {fwd_error:.2e}")
    
    # Print all metrics
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, list):
            print(f"  {k:30s} = {[f'{x:.4f}' for x in v]}")
        else:
            print(f"  {k:30s} = {v}")
    
    # Map to standard metric names
    std_metrics = {
        "PSNR": metrics["PSNR_abundance"],
        "CC": metrics["CC_abundance"],
        "RE": metrics["RE_abundance"],
        "RMSE": metrics["RMSE_abundance"],
        "mean_SAD_deg": metrics["mean_SAD_deg"],
        "method": method
    }
    
    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(std_metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), A_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), A_gt)
    
    # Visualize results
    visualize_results_internal(E_gt, E_rec, A_gt, A_rec, wavelengths, metrics, img_size,
                               os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    
    return std_metrics

def visualize_results_internal(E_gt, E_rec, A_gt, A_rec, wavelengths, metrics, img_size, save_path):
    """Create multi-panel figure: endmember spectra + abundance maps."""
    E_al, A_al, _ = align_endmembers(E_gt, E_rec, A_gt, A_rec)

    R = E_gt.shape[1]
    fig = plt.figure(figsize=(20, 12))

    # Top row: endmember spectra
    for i in range(R):
        ax = fig.add_subplot(3, R, i + 1)
        ax.plot(wavelengths, E_gt[:, i], 'b-', lw=1.5, label='GT')
        ax.plot(wavelengths, E_al[:, i], 'r--', lw=1.5, label='Recon')
        ax.set_title(f'Endmember {i+1}\nSAD={metrics["per_endmember_SAD_deg"][i]:.2f}°')
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel('Reflectance')

    # Middle row: GT abundances
    A_gt_imgs = A_gt.reshape(R, img_size, img_size)
    A_rec_imgs = A_al.reshape(R, img_size, img_size)
    for i in range(R):
        ax = fig.add_subplot(3, R, R + i + 1)
        ax.imshow(A_gt_imgs[i], cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'GT Abund. {i+1}')
        if i == 0:
            ax.set_ylabel('Ground Truth')

    # Bottom row: Reconstructed abundances
    for i in range(R):
        ax = fig.add_subplot(3, R, 2 * R + i + 1)
        ax.imshow(A_rec_imgs[i], cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'Recon {i+1}\nCC={metrics["per_endmember_CC"][i]:.3f}')
        if i == 0:
            ax.set_ylabel('Reconstructed')

    fig.suptitle(
        f"HySUPP — Hyperspectral Unmixing\n"
        f"CC={metrics['CC_abundance']:.4f} | SAD={metrics['mean_SAD_deg']:.2f}° | "
        f"PSNR={metrics['PSNR_abundance']:.1f} dB",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
