#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Electrical Resistivity Tomography (ERT) Inversion using pyGIMLi.

Benchmark inverse problem:
  Recover subsurface resistivity distribution from surface electrode array
  measurements using Gauss-Newton inversion with smoothness regularization.

Based on: Rücker et al., Comput. Geosci. 2017
Library:  pyGIMLi (gimli-org/pyGIMLi)
"""

import matplotlib
matplotlib.use('Agg')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pygimli as pg
import pygimli.meshtools as mt
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
        # Need 2D arrays for SSIM; reshape to approximate grid
        n = len(gt)
        side = int(np.ceil(np.sqrt(n)))
        # Pad arrays to make them square — use mean to avoid bias
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
        # Fallback: simple correlation-based metric
        gt_norm = (gt - np.mean(gt)) / (np.std(gt) + 1e-10)
        recon_norm = (recon - np.mean(recon)) / (np.std(recon) + 1e-10)
        return float(np.mean(gt_norm * recon_norm))


def compute_rmse(gt, recon):
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((gt - recon) ** 2)))


def interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=100, ny=50):
    """Interpolate cell-based values to a regular grid for visualization.

    Parameters
    ----------
    mesh : pg.Mesh
        The mesh containing cells.
    cell_values : array
        Values defined on mesh cells.
    x_range : tuple
        (xmin, xmax) of the grid.
    y_range : tuple
        (ymin, ymax) of the grid.
    nx, ny : int
        Grid resolution.

    Returns
    -------
    grid_values : 2D array (ny, nx)
    x_coords, y_coords : 1D arrays
    """
    from scipy.interpolate import griddata

    # Get cell centers
    cell_centers = np.array([[mesh.cell(i).center().x(),
                              mesh.cell(i).center().y()]
                             for i in range(mesh.cellCount())])

    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x_coords, y_coords)

    grid_values = griddata(cell_centers, np.array(cell_values),
                           (xx, yy), method='linear', fill_value=np.nan)

    return grid_values, x_coords, y_coords


def main():
    """Main function: synthetic ERT forward + inversion benchmark."""
    print("=" * 60)
    print("pyGIMLi ERT Inversion Benchmark")
    print("=" * 60)

    # ---- Create output directory ----
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # =========================================================================
    # 1) Build synthetic 2D subsurface model
    # =========================================================================
    print("\n[1/6] Creating synthetic subsurface model...")

    # Create the world geometry: 80m wide, 30m deep
    world = mt.createWorld(start=[-40, 0], end=[40, -30],
                           layers=[-5, -15], worldMarker=True)

    # Add resistivity anomalies:
    # Conductive block (low resistivity, e.g. clay body)
    block1 = mt.createRectangle(start=[-15, -3], end=[-5, -10],
                                marker=4, area=0.3)

    # Resistive block (high resistivity, e.g. rock)
    block2 = mt.createRectangle(start=[5, -5], end=[15, -12],
                                marker=5, area=0.3)

    # Conductive circular anomaly (e.g. contamination plume)
    circle = mt.createCircle(pos=[0, -18], radius=3, marker=6, area=0.3)

    # Merge into PLC
    geom = world + block1 + block2 + circle

    # =========================================================================
    # 2) Set up electrode array and measurement scheme
    # =========================================================================
    print("[2/6] Setting up electrode array (Dipole-dipole, 41 electrodes)...")

    # 41 electrodes from -20m to 20m (1m spacing)
    scheme = ert.createData(elecs=np.linspace(start=-20, stop=20, num=41),
                            schemeName='dd')  # Dipole-dipole

    print(f"   Number of electrodes: {scheme.sensorCount()}")
    print(f"   Number of measurements: {scheme.size()}")

    # Add electrode positions as nodes in the geometry for mesh refinement
    for p in scheme.sensors():
        geom.createNode(p)
        geom.createNode(p - [0, 0.1])

    # Create forward mesh
    mesh = mt.createMesh(geom, quality=34)
    print(f"   Forward mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")

    # =========================================================================
    # 3) Define resistivity distribution (ground truth)
    # =========================================================================
    # Region markers from world + anomalies:
    #   1 = top layer (0 to -5m)
    #   2 = middle layer (-5 to -15m)
    #   3 = bottom layer (-15 to -30m)
    #   4 = conductive block
    #   5 = resistive block
    #   6 = conductive circle
    rhomap = [
        [0, 100.0],   # Boundary/world default: 100 Ohm·m
        [1, 100.0],   # Top layer: 100 Ohm·m
        [2, 50.0],    # Middle layer: 50 Ohm·m
        [3, 200.0],   # Bottom layer: 200 Ohm·m
        [4, 10.0],    # Conductive block: 10 Ohm·m
        [5, 500.0],   # Resistive block: 500 Ohm·m
        [6, 5.0],     # Conductive circle: 5 Ohm·m
    ]

    # Build the full ground-truth resistivity vector on the mesh
    gt_res = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)
    gt_res_np = np.array(gt_res)
    print(f"   GT resistivity range: [{gt_res_np.min():.1f}, {gt_res_np.max():.1f}] Ohm·m")

    # =========================================================================
    # 4) Forward simulation: generate synthetic apparent resistivity data
    # =========================================================================
    print("[3/6] Running forward simulation (adding 3% noise + 1µV)...")

    data = ert.simulate(mesh, scheme=scheme, res=rhomap,
                        noiseLevel=0.03, noiseAbs=1e-6, seed=42)

    # Remove any negative apparent resistivities (can occur from noise)
    n_before = data.size()
    data.remove(data['rhoa'] < 0)
    n_after = data.size()
    if n_before != n_after:
        print(f"   Removed {n_before - n_after} negative rhoa values")

    print(f"   Simulated data points: {data.size()}")
    print(f"   Apparent resistivity range: [{min(data['rhoa']):.2f}, {max(data['rhoa']):.2f}] Ohm·m")

    # Save data to file for the manager
    data_file = os.path.join(results_dir, 'ert_data.dat')
    data.save(data_file)

    # =========================================================================
    # 5) Inversion: Gauss-Newton with smoothness regularization
    # =========================================================================
    print("[4/6] Running ERT inversion (Gauss-Newton, lambda=1)...")

    mgr = ert.ERTManager(data)

    # Run inversion with low regularization for maximum detail recovery
    # - lam=1: balanced regularization to avoid over-smoothing
    # - zWeight=0.3: reduce vertical smoothing for better depth resolution
    inv_model = mgr.invert(lam=1, zWeight=0.3, verbose=True)

    chi2 = mgr.inv.chi2()
    print(f"   Inversion chi² = {chi2:.3f}")
    print(f"   Inversion model size: {len(inv_model)}")

    # Get the parametric domain (inversion domain)
    pd = mgr.paraDomain
    inv_model_pd = mgr.paraModel(inv_model)

    print(f"   Para domain: {pd.cellCount()} cells")
    print(f"   Inverted resistivity range: [{min(inv_model_pd):.2f}, {max(inv_model_pd):.2f}] Ohm·m")

    # =========================================================================
    # 6) Interpolate both GT and reconstruction onto the same regular grid
    # =========================================================================
    print("[5/6] Computing metrics...")

    # --- Cell-based comparison on para domain ---
    # Map GT resistivity to para-domain cell centers for accurate comparison
    from scipy.interpolate import griddata as _griddata

    # Get para-domain cell centers
    pd_centers = np.array([[pd.cell(i).center().x(),
                            pd.cell(i).center().y()]
                           for i in range(pd.cellCount())])

    # Get forward mesh cell centers and GT values
    fwd_centers = np.array([[mesh.cell(i).center().x(),
                             mesh.cell(i).center().y()]
                            for i in range(mesh.cellCount())])

    # Interpolate GT onto para-domain cell centers (nearest to preserve blocky structure)
    gt_on_pd = _griddata(fwd_centers, gt_res_np, pd_centers,
                         method='nearest')

    inv_on_pd = np.array(inv_model_pd)

    # Work in log space for metrics (resistivity varies over orders of magnitude)
    gt_log_cells = np.log10(gt_on_pd)
    inv_log_cells = np.log10(inv_on_pd)

    psnr_val = compute_psnr(gt_log_cells, inv_log_cells)
    rmse_val = compute_rmse(gt_log_cells, inv_log_cells)

    # Also compute linear-space metrics
    psnr_lin = compute_psnr(gt_on_pd, inv_on_pd)
    rmse_lin = compute_rmse(gt_on_pd, inv_on_pd)

    # --- Grid interpolation for visualization ---
    xmin, xmax = pd.xmin(), pd.xmax()
    ymin, ymax = pd.ymin(), pd.ymax()
    nx, ny = 120, 60

    # Interpolate ground truth onto the same regular grid
    gt_grid, x_coords, y_coords = interpolate_to_grid(
        mesh, gt_res_np, (xmin, xmax), (ymin, ymax), nx, ny)

    # Interpolate inversion result onto the regular grid
    inv_grid, _, _ = interpolate_to_grid(
        pd, np.array(inv_model_pd), (xmin, xmax), (ymin, ymax), nx, ny)

    # Compute SSIM on the 2D grid (more natural for spatial data)
    # Fill NaN with mean to avoid zero-padding bias
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

    # =========================================================================
    # 7) Save metrics
    # =========================================================================
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

    # =========================================================================
    # 8) Save numpy arrays
    # =========================================================================
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_grid)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), inv_grid)
    print(f"   Arrays saved to {results_dir}/")

    # Also save in sandbox root for website_assets copy
    sandbox_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(sandbox_dir, 'gt_output.npy'), gt_grid)
    np.save(os.path.join(sandbox_dir, 'recon_output.npy'), inv_grid)
    # Save metrics to sandbox root
    with open(os.path.join(sandbox_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # =========================================================================
    # 9) Visualization
    # =========================================================================
    print("[6/6] Creating visualization...")

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Color scale (log)
    vmin, vmax = 5, 500
    cmap = 'Spectral_r'

    # --- (a) Ground Truth ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(x_coords, y_coords, gt_grid,
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, shading='auto')
    ax1.set_title('(a) Ground Truth Resistivity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_aspect('equal')
    cb1 = plt.colorbar(im1, ax=ax1, label='Resistivity (Ω·m)')

    # --- (b) Pseudosection of apparent resistivity ---
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        ert.showERTData(data, ax=ax2, cMap=cmap)
        ax2.set_title('(b) Apparent Resistivity Pseudosection', fontsize=13,
                      fontweight='bold')
    except Exception as e:
        print(f"   Pseudosection plot warning: {e}")
        # Fallback: simple plot of data
        ax2.scatter(range(data.size()), data['rhoa'], c=data['rhoa'],
                    cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                    s=5)
        ax2.set_title('(b) Apparent Resistivity Data', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Measurement #')
        ax2.set_ylabel('Apparent Resistivity (Ω·m)')

    # --- (c) Reconstruction ---
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(x_coords, y_coords, inv_grid,
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, shading='auto')
    ax3.set_title('(c) Reconstructed Resistivity', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_aspect('equal')
    cb3 = plt.colorbar(im3, ax=ax3, label='Resistivity (Ω·m)')

    # --- (d) Error map (log space) ---
    ax4 = fig.add_subplot(gs[1, 1])
    error_map = np.abs(np.log10(inv_grid) - np.log10(gt_grid))
    im4 = ax4.pcolormesh(x_coords, y_coords, error_map,
                         cmap='hot_r', shading='auto', vmin=0, vmax=1.5)
    ax4.set_title('(d) Log₁₀ Absolute Error', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_aspect('equal')
    cb4 = plt.colorbar(im4, ax=ax4, label='|log₁₀(recon) - log₁₀(GT)|')

    # --- (e) Cross-section comparison ---
    ax5 = fig.add_subplot(gs[2, 0])
    mid_y_idx = ny // 3  # roughly at ~1/3 depth
    depth_val = y_coords[mid_y_idx]
    ax5.semilogy(x_coords, gt_grid[mid_y_idx, :], 'b-', linewidth=2, label='Ground Truth')
    ax5.semilogy(x_coords, inv_grid[mid_y_idx, :], 'r--', linewidth=2, label='Reconstruction')
    ax5.set_title(f'(e) Horizontal Profile at depth={depth_val:.1f}m', fontsize=13,
                  fontweight='bold')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('Resistivity (Ω·m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- (f) Metrics summary ---
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
    # Also save to sandbox root as vis_result.png
    fig_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_result.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Figure saved to {fig_path}")

    print("\n" + "=" * 60)
    print("DONE. All outputs saved to:", results_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
