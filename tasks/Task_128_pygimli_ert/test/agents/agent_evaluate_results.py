import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from scipy.interpolate import griddata

from pygimli.physics import ert

def compute_psnr(gt, recon):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float('inf')
    data_range = np.max(gt) - np.min(gt)
    return 20.0 * np.log10(data_range / np.sqrt(mse))

def compute_ssim(gt, recon):
    """Compute SSIM between two 1D arrays (flattened cell-based)."""
    try:
        from skimage.metrics import structural_similarity
        n = len(gt)
        side = int(np.ceil(np.sqrt(n)))
        gt_mean = np.mean(gt)
        recon_mean = np.mean(recon)
        gt_pad = np.full(side * side, gt_mean)
        recon_pad = np.full(side * side, recon_mean)
        gt_pad[:n] = gt
        recon_pad[:n] = recon
        gt_2d = gt_pad.reshape(side, side)
        recon_2d = recon_pad.reshape(side, side)
        data_range = np.max(gt) - np.min(gt)
        win_size = min(7, side)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        return structural_similarity(gt_2d, recon_2d, data_range=data_range,
                                     win_size=win_size)
    except Exception as e:
        print(f"SSIM computation warning: {e}")
        gt_norm = (gt - np.mean(gt)) / (np.std(gt) + 1e-10)
        recon_norm = (recon - np.mean(recon)) / (np.std(recon) + 1e-10)
        return float(np.mean(gt_norm * recon_norm))

def compute_rmse(gt, recon):
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((gt - recon) ** 2)))

def interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=100, ny=50):
    """Interpolate cell-based values to a regular grid for visualization."""
    cell_centers = np.array([[mesh.cell(i).center().x(),
                              mesh.cell(i).center().y()]
                             for i in range(mesh.cellCount())])

    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x_coords, y_coords)

    grid_values = griddata(cell_centers, np.array(cell_values),
                           (xx, yy), method='linear', fill_value=np.nan)

    return grid_values, x_coords, y_coords

def evaluate_results(inversion_data):
    """
    Compute metrics and create visualization.
    
    Parameters
    ----------
    inversion_data : dict
        Output from run_inversion
    
    Returns
    -------
    dict containing all metrics
    """
    print("[5/6] Computing metrics...")

    mesh = inversion_data['mesh']
    scheme = inversion_data['scheme']
    data = inversion_data['data']
    gt_res_np = inversion_data['gt_res_np']
    inv_model_pd = inversion_data['inv_model_pd']
    pd = inversion_data['pd']
    chi2 = inversion_data['chi2']
    results_dir = inversion_data['results_dir']

    # Get para-domain cell centers
    pd_centers = np.array([[pd.cell(i).center().x(),
                            pd.cell(i).center().y()]
                           for i in range(pd.cellCount())])

    # Get forward mesh cell centers
    fwd_centers = np.array([[mesh.cell(i).center().x(),
                             mesh.cell(i).center().y()]
                            for i in range(mesh.cellCount())])

    # Interpolate GT onto para-domain cell centers
    gt_on_pd = griddata(fwd_centers, gt_res_np, pd_centers, method='nearest')
    inv_on_pd = np.array(inv_model_pd)

    # Work in log space
    gt_log_cells = np.log10(gt_on_pd)
    inv_log_cells = np.log10(inv_on_pd)

    psnr_val = compute_psnr(gt_log_cells, inv_log_cells)
    rmse_val = compute_rmse(gt_log_cells, inv_log_cells)

    psnr_lin = compute_psnr(gt_on_pd, inv_on_pd)
    rmse_lin = compute_rmse(gt_on_pd, inv_on_pd)

    # Grid interpolation for visualization
    xmin, xmax = pd.xmin(), pd.xmax()
    ymin, ymax = pd.ymin(), pd.ymax()
    nx, ny = 120, 60

    gt_grid, x_coords, y_coords = interpolate_to_grid(
        mesh, gt_res_np, (xmin, xmax), (ymin, ymax), nx, ny)

    inv_grid, _, _ = interpolate_to_grid(
        pd, np.array(inv_model_pd), (xmin, xmax), (ymin, ymax), nx, ny)

    # Compute SSIM on the 2D grid
    gt_grid_log = np.log10(np.where(np.isnan(gt_grid), 1.0, gt_grid))
    inv_grid_log = np.log10(np.where(np.isnan(inv_grid), 1.0, inv_grid))
    valid_mask = ~(np.isnan(gt_grid) | np.isnan(inv_grid))
    gt_grid_log_filled = gt_grid_log.copy()
    inv_grid_log_filled = inv_grid_log.copy()
    gt_grid_log_filled[~valid_mask] = np.mean(gt_grid_log[valid_mask])
    inv_grid_log_filled[~valid_mask] = np.mean(inv_grid_log[valid_mask])
    
    try:
        from skimage.metrics import structural_similarity
        data_range = np.max(gt_grid_log[valid_mask]) - np.min(gt_grid_log[valid_mask])
        ssim_val = structural_similarity(gt_grid_log_filled, inv_grid_log_filled,
                                         data_range=data_range, win_size=7)
    except Exception as e:
        print(f"   SSIM grid computation warning: {e}")
        ssim_val = compute_ssim(gt_log_cells, inv_log_cells)

    print(f"\n   === Metrics (log10 resistivity) ===")
    print(f"   PSNR  = {psnr_val:.2f} dB")
    print(f"   SSIM  = {ssim_val:.4f}")
    print(f"   RMSE  = {rmse_val:.4f}")
    print(f"\n   === Metrics (linear resistivity) ===")
    print(f"   PSNR  = {psnr_lin:.2f} dB")
    print(f"   RMSE  = {rmse_lin:.2f} Ohm·m")
    print(f"   Chi²  = {chi2:.3f}")

    # Save metrics
    metrics = {
        "PSNR_log_dB": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "RMSE_log": round(rmse_val, 4),
        "PSNR_linear_dB": round(psnr_lin, 2),
        "RMSE_linear_ohm_m": round(rmse_lin, 2),
        "chi2": round(chi2, 3),
        "num_electrodes": scheme.sensorCount(),
        "num_measurements": data.size(),
        "method": "Gauss-Newton with smoothness regularization",
        "scheme": "Dipole-dipole",
        "lambda": 1,
        "noise_level": 0.03,
    }

    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   Metrics saved to {metrics_path}")

    # Save numpy arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_grid)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), inv_grid)
    print(f"   Arrays saved to {results_dir}/")

    sandbox_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(sandbox_dir, 'gt_output.npy'), gt_grid)
    np.save(os.path.join(sandbox_dir, 'recon_output.npy'), inv_grid)
    with open(os.path.join(sandbox_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Visualization
    print("[6/6] Creating visualization...")

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    vmin, vmax = 5, 500
    cmap = 'Spectral_r'

    # (a) Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(x_coords, y_coords, gt_grid,
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, shading='auto')
    ax1.set_title('(a) Ground Truth Resistivity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Resistivity (Ω·m)')

    # (b) Pseudosection
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        ert.showERTData(data, ax=ax2, cMap=cmap)
        ax2.set_title('(b) Apparent Resistivity Pseudosection', fontsize=13,
                      fontweight='bold')
    except Exception as e:
        print(f"   Pseudosection plot warning: {e}")
        ax2.scatter(range(data.size()), data['rhoa'], c=data['rhoa'],
                    cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                    s=5)
        ax2.set_title('(b) Apparent Resistivity Data', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Measurement #')
        ax2.set_ylabel('Apparent Resistivity (Ω·m)')

    # (c) Reconstruction
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(x_coords, y_coords, inv_grid,
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, shading='auto')
    ax3.set_title('(c) Reconstructed Resistivity', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='Resistivity (Ω·m)')

    # (d) Error map
    ax4 = fig.add_subplot(gs[1, 1])
    error_map = np.abs(np.log10(inv_grid) - np.log10(gt_grid))
    im4 = ax4.pcolormesh(x_coords, y_coords, error_map,
                         cmap='hot_r', shading='auto', vmin=0, vmax=1.5)
    ax4.set_title('(d) Log₁₀ Absolute Error', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='|log₁₀(recon) - log₁₀(GT)|')

    # (e) Cross-section comparison
    ax5 = fig.add_subplot(gs[2, 0])
    mid_y_idx = ny // 3
    depth_val = y_coords[mid_y_idx]
    ax5.semilogy(x_coords, gt_grid[mid_y_idx, :], 'b-', linewidth=2, label='Ground Truth')
    ax5.semilogy(x_coords, inv_grid[mid_y_idx, :], 'r--', linewidth=2, label='Reconstruction')
    ax5.set_title(f'(e) Horizontal Profile at depth={depth_val:.1f}m', fontsize=13,
                  fontweight='bold')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('Resistivity (Ω·m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # (f) Metrics summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = (
        f"ERT Inversion Benchmark\n"
        f"{'─' * 35}\n"
        f"Method: Gauss-Newton + smoothness reg.\n"
        f"Scheme: Dipole-dipole, {scheme.sensorCount()} electrodes\n"
        f"Measurements: {data.size()}\n"
        f"Noise: 3% + 1µV\n"
        f"Lambda (regularization): 1\n"
        f"{'─' * 35}\n"
        f"PSNR (log₁₀):  {psnr_val:.2f} dB\n"
        f"SSIM (log₁₀):  {ssim_val:.4f}\n"
        f"RMSE (log₁₀):  {rmse_val:.4f}\n"
        f"PSNR (linear):  {psnr_lin:.2f} dB\n"
        f"RMSE (linear):  {rmse_lin:.2f} Ω·m\n"
        f"Chi²:  {chi2:.3f}\n"
    )
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('pyGIMLi ERT: Subsurface Resistivity Reconstruction',
                 fontsize=15, fontweight='bold', y=0.98)

    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    fig_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_result.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Figure saved to {fig_path}")

    print("\n" + "=" * 60)
    print("DONE. All outputs saved to:", results_dir)
    print("=" * 60)

    return metrics
