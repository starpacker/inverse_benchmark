import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

GRAV_CONST = 6.674e-3

def prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho):
    """
    Compute vertical gravity component gz for a rectangular prism.
    
    Uses the analytical formula from Blakely (1996) / Nagy (1966).
    
    Parameters
    ----------
    x1, x2 : float  Prism x bounds
    y1, y2 : float  Prism y bounds
    z1, z2 : float  Prism z bounds (z positive downward in convention, but we use z negative for depth)
    xp, yp, zp : float  Observation point coordinates
    rho : float  Density contrast [g/cm³]
    
    Returns
    -------
    gz : float  Vertical gravity component [mGal]
    """
    # Shift coordinates relative to observation point
    dx = [x1 - xp, x2 - xp]
    dy = [y1 - yp, y2 - yp]
    dz = [z1 - zp, z2 - zp]
    
    gz = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = dx[i]
                y = dy[j]
                z = dz[k]
                r = np.sqrt(x**2 + y**2 + z**2)
                
                # Avoid singularities
                r = max(r, 1e-10)
                
                # Sign for the sum
                sign = (-1) ** (i + j + k)
                
                # Compute terms
                term1 = 0.0
                term2 = 0.0
                term3 = 0.0
                
                # x * ln(y + r)
                if abs(y + r) > 1e-10:
                    term1 = x * np.log(y + r)
                
                # y * ln(x + r)
                if abs(x + r) > 1e-10:
                    term2 = y * np.log(x + r)
                
                # z * arctan(xy / (zr))
                denom = z * r
                if abs(denom) > 1e-10:
                    term3 = z * np.arctan2(x * y, denom)
                
                gz += sign * (term1 + term2 - term3)
    
    # Convert to mGal (G in appropriate units)
    # G = 6.674e-11 m³/(kg·s²), density in g/cm³ = 1000 kg/m³
    # 1 mGal = 1e-5 m/s²
    # gz = G * rho * integral, with rho in g/cm³
    # Factor: 6.674e-11 * 1000 * 1e5 = 6.674e-3
    gz *= GRAV_CONST * rho
    
    return gz

def forward_operator(model, mesh_info, rx_locs):
    """
    Gravity forward simulation using prism formula.
    
    Computes vertical gravity component gz at each receiver location
    by summing contributions from all mesh cells (treated as uniform
    density prisms).
    
    Parameters
    ----------
    model : np.ndarray  Density contrast vector (g/cm³), shape (n_cells,)
    mesh_info : dict    Mesh information
    rx_locs : np.ndarray  Receiver locations, shape (n_rx, 3)
    
    Returns
    -------
    d_pred : np.ndarray  Predicted gravity anomaly [mGal], shape (n_rx,)
    """
    n_rx = rx_locs.shape[0]
    n_cells = mesh_info['n_cells']
    cc = mesh_info['cell_centers']
    
    # Half cell sizes
    dx = mesh_info['hx'][0] / 2
    dy = mesh_info['hy'][0] / 2
    dz = mesh_info['hz'][0] / 2
    
    d_pred = np.zeros(n_rx)
    
    # Only process cells with non-zero density for efficiency
    active_cells = np.where(model != 0)[0]
    
    for i_rx in range(n_rx):
        xp, yp, zp = rx_locs[i_rx]
        gz_total = 0.0
        
        for i_cell in active_cells:
            xc, yc, zc = cc[i_cell]
            rho = model[i_cell]
            
            # Prism bounds
            x1, x2 = xc - dx, xc + dx
            y1, y2 = yc - dy, yc + dy
            z1, z2 = zc - dz, zc + dz
            
            gz_total += prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho)
        
        d_pred[i_rx] = gz_total
    
    return d_pred

def evaluate_results(data_dict, model_rec):
    """
    Compute inversion quality metrics and visualize results.
    
    Parameters
    ----------
    data_dict : dict  Contains mesh_info, model_gt, d_clean, rx_locs
    model_rec : np.ndarray  Recovered density model
    
    Returns
    -------
    metrics : dict  Quality metrics
    """
    mesh_info = data_dict['mesh_info']
    model_gt = data_dict['model_gt']
    d_clean = data_dict['d_clean']
    rx_locs = data_dict['rx_locs']
    d_noisy = data_dict['d_noisy']
    
    # Compute predicted data from recovered model
    d_rec = forward_operator(model_rec, mesh_info, rx_locs)
    
    # Model-space metrics (3D → reshape to evaluate per-layer)
    nx, ny, nz = mesh_info['shape_cells']
    gt_3d = model_gt.reshape((nx, ny, nz), order='F')
    rec_3d = model_rec.reshape((nx, ny, nz), order='F')

    # Take a horizontal slice at anomaly depth
    iz_anom = nz // 2
    gt_slice = gt_3d[:, :, iz_anom]
    rec_slice = rec_3d[:, :, iz_anom]

    data_range = gt_slice.max() - gt_slice.min()
    if data_range < 1e-12:
        data_range = 1.0

    mse = np.mean((gt_slice - rec_slice) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))
    
    # SSIM calculation (simplified version)
    def compute_ssim(im1, im2, data_range):
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        mu1 = np.mean(im1)
        mu2 = np.mean(im2)
        sigma1_sq = np.var(im1)
        sigma2_sq = np.var(im2)
        sigma12 = np.mean((im1 - mu1) * (im2 - mu2))
        
        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return num / den
    
    ssim_val = float(compute_ssim(gt_slice, rec_slice, data_range))
    
    # Correlation coefficients
    gt_flat = gt_slice.ravel()
    rec_flat = rec_slice.ravel()
    if np.std(gt_flat) > 1e-12 and np.std(rec_flat) > 1e-12:
        cc_slice = float(np.corrcoef(gt_flat, rec_flat)[0, 1])
    else:
        cc_slice = 0.0

    # Volume metrics
    if np.std(model_gt) > 1e-12 and np.std(model_rec) > 1e-12:
        cc_vol = float(np.corrcoef(model_gt, model_rec)[0, 1])
    else:
        cc_vol = 0.0
    
    re_vol = float(np.linalg.norm(model_gt - model_rec) /
                   max(np.linalg.norm(model_gt), 1e-12))

    # Data fit metrics
    residual = d_clean - d_rec
    rmse_data = float(np.sqrt(np.mean(residual ** 2)))
    if np.std(d_clean) > 1e-12 and np.std(d_rec) > 1e-12:
        cc_data = float(np.corrcoef(d_clean, d_rec)[0, 1])
    else:
        cc_data = 0.0

    metrics = {
        "PSNR_slice": psnr,
        "SSIM_slice": ssim_val,
        "CC_slice": cc_slice,
        "CC_volume": cc_vol,
        "RE_volume": re_vol,
        "RMSE_data_mGal": rmse_data,
        "CC_data": cc_data,
    }
    
    # Print metrics
    print("\n[EVAL] Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:25s} = {v:.6f}")
    
    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), model_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), model_gt)
    
    # Visualization
    visualize_results(mesh_info, model_gt, model_rec, rx_locs,
                      d_clean, d_noisy, d_rec, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    
    return metrics

def visualize_results(mesh_info, model_gt, model_rec, rx_locs,
                      d_clean, d_noisy, d_rec, metrics, save_path):
    """Create visualization of inversion results."""
    nx, ny, nz = mesh_info['shape_cells']
    gt_3d = model_gt.reshape((nx, ny, nz), order='F')
    rec_3d = model_rec.reshape((nx, ny, nz), order='F')

    iz = nz // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmax = max(np.abs(gt_3d).max(), 0.1)

    # (a) GT slice
    im = axes[0, 0].imshow(gt_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'(a) GT Density (z-slice {iz})')
    plt.colorbar(im, ax=axes[0, 0], label='Δρ [g/cm³]')

    # (b) Reconstructed slice
    im = axes[0, 1].imshow(rec_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 1].set_title('(b) Inversion Result')
    plt.colorbar(im, ax=axes[0, 1], label='Δρ [g/cm³]')

    # (c) Error
    err = gt_3d[:, :, iz] - rec_3d[:, :, iz]
    im = axes[0, 2].imshow(err.T, cmap='RdBu_r',
                            vmin=-vmax/2, vmax=vmax/2, origin='lower')
    axes[0, 2].set_title('(c) Error')
    plt.colorbar(im, ax=axes[0, 2], label='Δρ error')

    # (d) Observed data map
    n_rx = int(np.sqrt(len(d_clean)))
    if n_rx ** 2 == len(d_clean):
        d_map = d_clean.reshape(n_rx, n_rx)
        axes[1, 0].imshow(d_map, cmap='viridis', origin='lower')
    else:
        axes[1, 0].scatter(rx_locs[:, 0], rx_locs[:, 1],
                           c=d_clean, cmap='viridis', s=20)
    axes[1, 0].set_title('(d) Gravity Anomaly (GT)')

    # (e) Data fit
    axes[1, 1].plot(d_clean, d_rec, 'b.', ms=3)
    lims = [min(d_clean.min(), d_rec.min()),
            max(d_clean.max(), d_rec.max())]
    axes[1, 1].plot(lims, lims, 'k--', lw=0.5)
    axes[1, 1].set_xlabel('True g_z [mGal]')
    axes[1, 1].set_ylabel('Predicted g_z [mGal]')
    axes[1, 1].set_title(f'(e) Data Fit  CC={metrics["CC_data"]:.4f}')

    # (f) Depth profile
    axes[1, 2].plot(gt_3d[nx//2, ny//2, :], range(nz), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_3d[nx//2, ny//2, :], range(nz), 'r--', lw=2, label='Inv')
    axes[1, 2].set_xlabel('Δρ [g/cm³]')
    axes[1, 2].set_ylabel('Depth index')
    axes[1, 2].set_title('(f) Depth Profile')
    axes[1, 2].legend()
    axes[1, 2].invert_yaxis()

    fig.suptitle(
        f"Gravity Anomaly Inversion\n"
        f"PSNR={metrics['PSNR_slice']:.1f} dB  |  "
        f"SSIM={metrics['SSIM_slice']:.4f}  |  "
        f"CC_vol={metrics['CC_volume']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
