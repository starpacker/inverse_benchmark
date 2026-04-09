import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

def evaluate_results(data, inversion_result, results_dir):
    """
    Compute metrics, save results, and create visualizations.
    
    Args:
        data: dict from load_and_preprocess_data
        inversion_result: dict from run_inversion
        results_dir: directory to save results
    
    Returns:
        dict of metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    eps_gt = data['eps_gt']
    Ez_gt = data['Ez_gt']
    params = data['params']
    
    eps_opt = inversion_result['eps_opt']
    Ez_opt = inversion_result['Ez_opt']
    loss_history = inversion_result['loss_history']
    
    nx = params['nx']
    ny = params['ny']
    npml = params['npml']
    wg_width = params['wg_width']
    eps_min = params['eps_min']
    eps_max = params['eps_max']
    
    # Compute metrics
    print("\n[4] Computing metrics...")
    
    # Normalize permittivity to [0,1]
    eps_gt_n = (eps_gt - eps_min) / (eps_max - eps_min)
    eps_opt_n = (eps_opt - eps_min) / (eps_max - eps_min)
    
    # Structure metrics
    mse_s = float(np.mean((eps_gt_n - eps_opt_n)**2))
    psnr_s = float(10 * np.log10(1.0 / (mse_s + 1e-10)))
    ssim_s = float(ssim(eps_gt_n, eps_opt_n, data_range=1.0, win_size=3))
    cc_s = float(np.corrcoef(eps_gt_n.flatten(), eps_opt_n.flatten())[0, 1])
    
    # Field metrics
    Ez_gt_a = np.abs(Ez_gt)
    Ez_opt_a = np.abs(Ez_opt)
    Ez_gt_n2 = Ez_gt_a / (np.max(Ez_gt_a) + 1e-30)
    Ez_opt_n2 = Ez_opt_a / (np.max(Ez_opt_a) + 1e-30)
    
    cc_f = float(np.corrcoef(Ez_gt_n2.flatten(), Ez_opt_n2.flatten())[0, 1])
    mse_f = float(np.mean((Ez_gt_n2 - Ez_opt_n2)**2))
    psnr_f = float(10 * np.log10(1.0 / (mse_f + 1e-10)))
    ssim_f = float(ssim(Ez_gt_n2, Ez_opt_n2, data_range=1.0, win_size=3))
    
    # Transmission at a probe location (right side of waveguide)
    probe = np.zeros((nx, ny))
    cx = nx // 2
    hw = wg_width // 2
    probe_y = ny - npml - 2
    probe[cx - hw : cx + hw + 1, probe_y] = 1.0
    T_gt_val = float(np.sum(np.abs(Ez_gt * probe)**2))
    T_opt_val = float(np.sum(np.abs(Ez_opt * probe)**2))
    T_ratio = T_opt_val / (T_gt_val + 1e-30)
    
    metrics = {
        "PSNR": psnr_s,
        "SSIM": ssim_s,
        "structure_psnr_db": psnr_s,
        "structure_ssim": ssim_s,
        "structure_cc": cc_s,
        "structure_mse": mse_s,
        "field_psnr_db": psnr_f,
        "field_ssim": ssim_f,
        "field_cc": cc_f,
        "transmission_gt": T_gt_val,
        "transmission_opt": T_opt_val,
        "transmission_ratio": T_ratio,
    }
    
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.6f}")
    
    # Save outputs
    print("\n[5] Saving outputs...")
    np.save(os.path.join(results_dir, "gt_output.npy"), eps_gt)
    np.save(os.path.join(results_dir, "recon_output.npy"), eps_opt)
    np.save(os.path.join(results_dir, "gt_field.npy"), np.abs(Ez_gt))
    np.save(os.path.join(results_dir, "opt_field.npy"), np.abs(Ez_opt))
    np.save(os.path.join(results_dir, "loss_history.npy"), np.array(loss_history))
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path}")
    
    # Create visualization
    print("\n[6] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    eps_gt_d = (eps_gt - eps_min) / (eps_max - eps_min)
    eps_opt_d = (eps_opt - eps_min) / (eps_max - eps_min)
    
    # Panel 1: GT structure
    ax = axes[0, 0]
    im = ax.imshow(eps_gt_d.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('Ground Truth Dielectric Structure', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='Normalized eps', shrink=0.8)
    
    # Panel 2: Optimized structure
    ax = axes[0, 1]
    im = ax.imshow(eps_opt_d.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title(f'Optimized Structure (CC={metrics["structure_cc"]:.3f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='Normalized eps', shrink=0.8)
    
    # Panel 3: Field pattern (optimized)
    ax = axes[1, 0]
    Ez_opt_n = np.abs(Ez_opt) / (np.max(np.abs(Ez_opt)) + 1e-30)
    im = ax.imshow(Ez_opt_n.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'|Ez| Field (Optimized), Field CC={metrics["field_cc"]:.3f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='|Ez| (normalized)', shrink=0.8)
    
    # Panel 4: Error map
    ax = axes[1, 1]
    error = np.abs(eps_gt_d - eps_opt_d)
    im = ax.imshow(error.T, origin='lower', cmap='viridis', vmin=0)
    ax.set_title(f'|eps_GT - eps_opt| Error (PSNR={metrics["structure_psnr_db"]:.1f} dB)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='|Delta eps|', shrink=0.8)
    
    wavelength = 1.55e-6
    plt.suptitle('Task 158: Photonic Inverse Design (FDFD, ceviche)\n'
                 f'Grid: {nx}x{ny}, lambda={wavelength*1e9:.0f} nm, '
                 f'SSIM={metrics["structure_ssim"]:.3f}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    vis_path = os.path.join(results_dir, "vis_result.png")
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {vis_path}")
    
    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(loss_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Optimization Convergence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    conv_path = os.path.join(results_dir, "convergence.png")
    fig.savefig(conv_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {conv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Structure PSNR: {metrics['structure_psnr_db']:.2f} dB")
    print(f"  Structure SSIM: {metrics['structure_ssim']:.4f}")
    print(f"  Structure CC:   {metrics['structure_cc']:.4f}")
    print(f"  Field CC:       {metrics['field_cc']:.4f}")
    print(f"  Field SSIM:     {metrics['field_ssim']:.4f}")
    print(f"  Transmission ratio: {metrics['transmission_ratio']:.4f}")
    print("=" * 70)
    
    return metrics
