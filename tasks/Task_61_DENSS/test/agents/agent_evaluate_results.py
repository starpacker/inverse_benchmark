import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

from skimage.metrics import structural_similarity as ssim_fn

def forward_operator(density, voxel_size, n_q, q_max):
    """
    Compute 1D SAXS profile I(q) from 3D electron density.

    I(q) = spherical_average( |FFT{ρ(r)}|² )

    Parameters
    ----------
    density : ndarray
        3D electron density array.
    voxel_size : float
        Voxel size in Angstroms.
    n_q : int
        Number of q bins.
    q_max : float
        Maximum q value in inverse Angstroms.

    Returns
    -------
    q_bins : ndarray
        q values in inverse Angstroms.
    I_q : ndarray
        Scattering intensity (normalized).
    """
    N = density.shape[0]

    # 3D FFT
    F = fftshift(fftn(ifftshift(density)))
    I_3d = np.abs(F) ** 2

    # q-grid
    freq = fftfreq(N, d=voxel_size)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Radial average (spherical shells)
    q_bins = np.linspace(0.01, q_max, n_q)
    dq = q_bins[1] - q_bins[0]
    I_q = np.zeros(n_q)

    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq / 2) & (q_3d < qc + dq / 2)
        if mask.sum() > 0:
            I_q[i] = np.mean(I_3d[mask])

    # Normalise
    if I_q.max() > 0:
        I_q = I_q / I_q.max()

    return q_bins, I_q

def evaluate_results(density_gt, density_rec, I_clean, q, voxel_size, n_q, q_max, results_dir):
    """
    Compute reconstruction metrics and generate visualizations.

    Parameters
    ----------
    density_gt : ndarray
        Ground truth 3D electron density.
    density_rec : ndarray
        Reconstructed 3D electron density.
    I_clean : ndarray
        Clean I(q) values.
    q : ndarray
        q values.
    voxel_size : float
        Voxel size in Angstroms.
    n_q : int
        Number of q bins.
    q_max : float
        Maximum q value.
    results_dir : str
        Directory to save results.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    # Normalize
    gt = density_gt / max(density_gt.max(), 1e-12)
    rec = density_rec / max(density_rec.max(), 1e-12)

    # Ensure same shape
    s = min(gt.shape[0], rec.shape[0])
    gt = gt[:s, :s, :s]
    rec = rec[:s, :s, :s]

    # 3D CC
    cc_vol = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])
    re_vol = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    # Central slice metrics
    mid = s // 2
    gt_slice = gt[mid, :, :]
    rec_slice = rec[mid, :, :]
    dr = gt_slice.max() - gt_slice.min()
    if dr < 1e-12:
        dr = 1.0
    mse = np.mean((gt_slice - rec_slice) ** 2)
    psnr = float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_slice, rec_slice, data_range=dr))
    cc_slice = float(np.corrcoef(gt_slice.ravel(), rec_slice.ravel())[0, 1])

    # I(q) fit
    _, I_rec = forward_operator(rec * density_rec.max(), voxel_size, n_q, q_max)
    cc_Iq = float(np.corrcoef(I_clean, I_rec)[0, 1])

    metrics = {
        "PSNR_slice": psnr,
        "SSIM_slice": ssim_val,
        "CC_slice": cc_slice,
        "CC_volume": cc_vol,
        "RE_volume": re_vol,
        "CC_Iq": cc_Iq,
    }

    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), density_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), density_gt)

    # Generate visualization
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    
    # Load I_noisy for plotting (recompute with noise for visualization)
    rng = np.random.default_rng(42)
    I_noisy = I_clean * (1 + 0.001 * rng.standard_normal(len(I_clean)))
    I_noisy = np.maximum(I_noisy, 0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mid = density_gt.shape[0] // 2
    gt_n = density_gt / max(density_gt.max(), 1e-12)
    rec_n = density_rec / max(density_rec.max(), 1e-12)
    s = min(gt_n.shape[0], rec_n.shape[0])

    for i, (title, data) in enumerate([
        ('GT (z-slice)', gt_n[min(mid, s - 1)]),
        ('Recon (z-slice)', rec_n[min(mid, s - 1)]),
        ('Error', gt_n[min(mid, s - 1)] - rec_n[min(mid, s - 1)]),
    ]):
        axes[0, i].imshow(data, cmap='hot' if i < 2 else 'RdBu_r', origin='lower')
        axes[0, i].set_title(title)

    axes[1, 0].semilogy(q, I_clean, 'b-', lw=2, label='GT')
    axes[1, 0].semilogy(q, I_noisy, 'k.', ms=3, alpha=0.5, label='Noisy')
    axes[1, 0].set_xlabel('q [Å⁻¹]')
    axes[1, 0].set_ylabel('I(q)')
    axes[1, 0].set_title('SAXS Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gt_n[min(mid, s - 1), min(mid, s - 1), :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(rec_n[min(mid, s - 1), min(mid, s - 1), :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title('1D Line Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].text(0.5, 0.5, '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
                    transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=11,
                    family='monospace')
    axes[1, 2].set_title('Metrics')
    axes[1, 2].axis('off')

    fig.suptitle(f"DENSS — SAXS Electron Density Reconstruction\n"
                 f"PSNR={metrics['PSNR_slice']:.1f} dB  |  SSIM={metrics['SSIM_slice']:.4f}  |  "
                 f"CC={metrics['CC_volume']:.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics
