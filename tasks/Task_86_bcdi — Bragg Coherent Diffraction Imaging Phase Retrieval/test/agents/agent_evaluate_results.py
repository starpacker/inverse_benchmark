import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

import warnings

warnings.filterwarnings('ignore')

def evaluate_results(obj_true, obj_recon, support, intensity, errors, results_dir):
    """
    Evaluate phase retrieval quality, align phases, compute metrics, and visualize.
    
    Args:
        obj_true: ground truth 3D complex object
        obj_recon: reconstructed 3D complex object
        support: 3D binary support mask
        intensity: measured diffraction intensity
        errors: convergence history
        results_dir: directory to save results
        
    Returns:
        metrics: dictionary of evaluation metrics
        obj_aligned: phase-aligned reconstruction
    """
    # Phase alignment - resolve twin-image ambiguity and global phase offset
    candidates = [obj_recon, np.conj(obj_recon[::-1, ::-1, ::-1])]
    
    best_cc = -1
    best_obj = None
    
    for cand in candidates:
        mask = support > 0.5
        if np.sum(mask) == 0:
            continue
        
        # Compute optimal phase offset
        cross = np.sum(cand[mask] * np.conj(obj_true[mask]))
        phase_offset = np.angle(cross)
        cand_aligned = cand * np.exp(-1j * phase_offset)
        
        # Compute correlation
        cc = np.abs(np.sum(cand_aligned[mask] * np.conj(obj_true[mask]))) / \
             (np.sqrt(np.sum(np.abs(cand_aligned[mask])**2) * np.sum(np.abs(obj_true[mask])**2)))
        
        if cc > best_cc:
            best_cc = cc
            best_obj = cand_aligned
    
    obj_aligned = best_obj
    print(f"[POST] Alignment CC = {best_cc:.6f}")
    
    # Compute metrics
    mask = support > 0.5
    
    # Amplitude metrics
    amp_true = np.abs(obj_true[mask])
    amp_recon = np.abs(obj_aligned[mask])
    
    amp_mse = np.mean((amp_true - amp_recon)**2)
    amp_range = amp_true.max() - amp_true.min()
    psnr_amp = 10 * np.log10(amp_range**2 / amp_mse) if amp_mse > 0 else float('inf')
    
    cc_amp = np.corrcoef(amp_true, amp_recon)[0, 1]
    
    # Phase metrics (within support)
    phase_true = np.angle(obj_true[mask])
    phase_recon = np.angle(obj_aligned[mask])
    
    # Phase difference (wrapped)
    phase_diff = np.angle(np.exp(1j * (phase_true - phase_recon)))
    phase_rmse = np.sqrt(np.mean(phase_diff**2))
    
    # Complex-valued correlation
    cc_complex = np.abs(np.sum(obj_aligned[mask] * np.conj(obj_true[mask]))) / \
                 np.sqrt(np.sum(np.abs(obj_aligned[mask])**2) * np.sum(np.abs(obj_true[mask])**2))
    
    # R-factor (crystallographic)
    ft_true = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_true))))
    ft_recon = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_aligned))))
    r_factor = np.sum(np.abs(ft_true - ft_recon)) / np.sum(ft_true)
    
    # Amplitude PSNR for the full 3D volume
    amp_full_true = np.abs(obj_true)
    amp_full_recon = np.abs(obj_aligned)
    mse_full = np.mean((amp_full_true - amp_full_recon)**2)
    range_full = amp_full_true.max() - amp_full_true.min()
    psnr_full = 10 * np.log10(range_full**2 / mse_full) if mse_full > 0 else float('inf')
    
    metrics = {
        'psnr_amplitude_support': float(psnr_amp),
        'psnr_amplitude_full': float(psnr_full),
        'cc_amplitude': float(cc_amp),
        'cc_complex': float(cc_complex),
        'phase_rmse_rad': float(phase_rmse),
        'r_factor': float(r_factor),
    }
    
    for k, v in metrics.items():
        print(f"[EVAL] {k} = {v:.6f}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "input.npy"), intensity)
    np.save(os.path.join(results_dir, "ground_truth.npy"), obj_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), obj_aligned)
    print(f"[SAVE] Input shape: {intensity.shape}")
    print(f"[SAVE] GT shape: {obj_true.shape}")
    print(f"[SAVE] Recon shape: {obj_aligned.shape}")
    
    # Visualization
    N = obj_true.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    mid = N // 2
    
    # (a) True amplitude - central slice
    ax = axes[0, 0]
    im = ax.imshow(np.abs(obj_true[:, :, mid]), cmap='hot', origin='lower')
    ax.set_title('True Amplitude (z-mid)')
    plt.colorbar(im, ax=ax)
    
    # (b) True phase - central slice
    ax = axes[0, 1]
    phase_true_slice = np.angle(obj_true[:, :, mid])
    phase_true_masked = np.where(support[:, :, mid] > 0.5, phase_true_slice, np.nan)
    im2 = ax.imshow(phase_true_masked, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title('True Phase (z-mid)')
    plt.colorbar(im2, ax=ax, label='rad')
    
    # (c) Diffraction intensity - central slice (log scale)
    ax = axes[0, 2]
    im3 = ax.imshow(np.log10(intensity[:, :, mid] + 1), cmap='viridis', origin='lower')
    ax.set_title('log₁₀(Intensity+1) (z-mid)')
    plt.colorbar(im3, ax=ax)
    
    # (d) Reconstructed amplitude
    ax = axes[0, 3]
    im4 = ax.imshow(np.abs(obj_aligned[:, :, mid]), cmap='hot', origin='lower')
    ax.set_title('Recon Amplitude (z-mid)')
    plt.colorbar(im4, ax=ax)
    
    # (e) Reconstructed phase
    ax = axes[1, 0]
    phase_recon_slice = np.angle(obj_aligned[:, :, mid])
    phase_recon_masked = np.where(support[:, :, mid] > 0.5, phase_recon_slice, np.nan)
    im5 = ax.imshow(phase_recon_masked, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Recon Phase (z-mid)')
    plt.colorbar(im5, ax=ax, label='rad')
    
    # (f) Phase error
    ax = axes[1, 1]
    phase_diff_slice = np.angle(np.exp(1j * (phase_true_slice - phase_recon_slice)))
    phase_diff_masked = np.where(support[:, :, mid] > 0.5, phase_diff_slice, np.nan)
    im6 = ax.imshow(phase_diff_masked, cmap='seismic', origin='lower', vmin=-0.5, vmax=0.5)
    ax.set_title('Phase Error (z-mid)')
    plt.colorbar(im6, ax=ax, label='rad')
    
    # (g) Convergence
    ax = axes[1, 2]
    ax.semilogy(errors)
    ax.set_xlabel('Iteration checkpoint')
    ax.set_ylabel('R-factor²')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)
    
    # (h) Amplitude scatter
    ax = axes[1, 3]
    ax.scatter(np.abs(obj_true[mask]), np.abs(obj_aligned[mask]),
               s=1, alpha=0.3, c='steelblue')
    lim = max(np.abs(obj_true[mask]).max(), np.abs(obj_aligned[mask]).max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', lw=2)
    ax.set_xlabel('True |ρ|')
    ax.set_ylabel('Recon |ρ|')
    ax.set_title(f'Amplitude (CC={metrics["cc_amplitude"]:.4f})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"bcdi — Bragg CDI Phase Retrieval (HIO+ER)\n"
        f"PSNR={metrics['psnr_amplitude_full']:.2f} dB | "
        f"CC_complex={metrics['cc_complex']:.4f} | "
        f"Phase RMSE={metrics['phase_rmse_rad']:.4f} rad | "
        f"R-factor={metrics['r_factor']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {vis_path}")
    
    return metrics, obj_aligned
