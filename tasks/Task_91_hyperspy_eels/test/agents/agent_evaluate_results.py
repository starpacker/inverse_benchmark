import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(ground_truth, reconstruction, measured, zlp, energy_axis, results_dir):
    """
    Compute quality metrics and generate visualizations.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground truth SSD.
    reconstruction : ndarray
        Reconstructed SSD.
    measured : ndarray
        Measured EELS spectrum.
    zlp : ndarray
        Zero-loss peak.
    energy_axis : ndarray
        Energy axis in eV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, RMSE, CC, relative_error.
    """
    # Define region of interest (ROI: 2–80 eV)
    roi = (energy_axis >= 2.0) & (energy_axis <= 80.0)
    gt_roi = ground_truth[roi].astype(np.float64)
    rec_roi = reconstruction[roi].astype(np.float64)
    
    # Compute MSE and RMSE
    mse = np.mean((gt_roi - rec_roi)**2)
    rmse = np.sqrt(mse)
    
    # Compute PSNR
    data_range = np.max(gt_roi) - np.min(gt_roi)
    if data_range > 0 and rmse > 0:
        psnr = 20.0 * np.log10(data_range / rmse)
    else:
        psnr = float('inf')
    
    # Compute correlation coefficient
    gt_c = gt_roi - np.mean(gt_roi)
    rec_c = rec_roi - np.mean(rec_roi)
    denom = np.sqrt(np.sum(gt_c**2) * np.sum(rec_c**2))
    cc = float(np.sum(gt_c * rec_c) / denom) if denom > 0 else 0.0
    
    # Compute relative error
    gt_norm = np.linalg.norm(gt_roi)
    rel_err = float(np.linalg.norm(gt_roi - rec_roi) / gt_norm) if gt_norm > 0 else float('inf')
    
    metrics = {
        "PSNR": float(np.round(psnr, 4)),
        "RMSE": float(np.round(rmse, 6)),
        "CC": float(np.round(cc, 6)),
        "relative_error": float(np.round(rel_err, 6))
    }
    
    # Save metrics to JSON
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save numpy arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), ground_truth)
    np.save(os.path.join(results_dir, "reconstruction.npy"), reconstruction)
    np.save(os.path.join(results_dir, "input_measurement.npy"), measured)
    np.save(os.path.join(results_dir, "energy_axis.npy"), energy_axis)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             gridspec_kw={'height_ratios': [1.2, 1.2, 0.8]})
    xl = [0, 80]
    
    # Panel 1: Measured EELS
    ax = axes[0]
    ax.plot(energy_axis, measured, 'b-', lw=1.0, alpha=0.8,
            label='Measured EELS (multiple scattering)')
    scale = 0.3 * np.max(measured) / (np.max(zlp) + 1e-30)
    ax.plot(energy_axis, zlp * scale, 'g--', lw=1.0, alpha=0.7,
            label='Zero-Loss Peak (scaled)')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('EELS Measurement with Multiple Scattering',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: GT vs Reconstructed SSD
    ax = axes[1]
    ax.plot(energy_axis, ground_truth, 'k-', lw=2.0, label='Ground Truth SSD')
    ax.plot(energy_axis, reconstruction, 'r--', lw=1.5, alpha=0.9,
            label='Reconstructed SSD (Fourier-Log)')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title(
        f'Single Scattering Distribution Recovery\n'
        f'PSNR = {metrics["PSNR"]:.2f} dB | CC = {metrics["CC"]:.4f} | '
        f'RMSE = {metrics["RMSE"]:.4e}',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Residual
    ax = axes[2]
    residual = ground_truth - reconstruction
    ax.plot(energy_axis, residual, 'purple', lw=1.0, alpha=0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.fill_between(energy_axis, residual, alpha=0.2, color='purple')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Residual (GT − Reconstruction)', fontsize=13, fontweight='bold')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
