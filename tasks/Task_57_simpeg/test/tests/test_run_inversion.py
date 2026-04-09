import sys
import os
import dill
import numpy as np
import traceback
import json

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================================
# Inject the Referee (Evaluation Logic) from Reference B
# ============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRAV_CONST = 6.674e-3

def prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho):
    """
    Compute vertical gravity component gz for a rectangular prism.
    """
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
                
                r = max(r, 1e-10)
                sign = (-1) ** (i + j + k)
                
                term1 = 0.0
                term2 = 0.0
                term3 = 0.0
                
                if abs(y + r) > 1e-10:
                    term1 = x * np.log(y + r)
                
                if abs(x + r) > 1e-10:
                    term2 = y * np.log(x + r)
                
                denom = z * r
                if abs(denom) > 1e-10:
                    term3 = z * np.arctan2(x * y, denom)
                
                gz += sign * (term1 + term2 - term3)
    
    gz *= GRAV_CONST * rho
    return gz

def forward_operator(model, mesh_info, rx_locs):
    """
    Gravity forward simulation using prism formula.
    """
    n_rx = rx_locs.shape[0]
    n_cells = mesh_info['n_cells']
    cc = mesh_info['cell_centers']
    
    dx = mesh_info['hx'][0] / 2
    dy = mesh_info['hy'][0] / 2
    dz = mesh_info['hz'][0] / 2
    
    d_pred = np.zeros(n_rx)
    
    active_cells = np.where(model != 0)[0]
    
    for i_rx in range(n_rx):
        xp, yp, zp = rx_locs[i_rx]
        gz_total = 0.0
        
        for i_cell in active_cells:
            xc, yc, zc = cc[i_cell]
            rho = model[i_cell]
            
            x1, x2 = xc - dx, xc + dx
            y1, y2 = yc - dy, yc + dy
            z1, z2 = zc - dz, zc + dz
            
            gz_total += prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho)
        
        d_pred[i_rx] = gz_total
    
    return d_pred

def evaluate_results(data_dict, model_rec):
    """
    Compute inversion quality metrics and visualize results.
    """
    mesh_info = data_dict['mesh_info']
    model_gt = data_dict['model_gt']
    d_clean = data_dict['d_clean']
    rx_locs = data_dict['rx_locs']
    d_noisy = data_dict['d_noisy']
    
    d_rec = forward_operator(model_rec, mesh_info, rx_locs)
    
    nx, ny, nz = mesh_info['shape_cells']
    gt_3d = model_gt.reshape((nx, ny, nz), order='F')
    rec_3d = model_rec.reshape((nx, ny, nz), order='F')

    iz_anom = nz // 2
    gt_slice = gt_3d[:, :, iz_anom]
    rec_slice = rec_3d[:, :, iz_anom]

    data_range = gt_slice.max() - gt_slice.min()
    if data_range < 1e-12:
        data_range = 1.0

    mse = np.mean((gt_slice - rec_slice) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))
    
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
    
    gt_flat = gt_slice.ravel()
    rec_flat = rec_slice.ravel()
    if np.std(gt_flat) > 1e-12 and np.std(rec_flat) > 1e-12:
        cc_slice = float(np.corrcoef(gt_flat, rec_flat)[0, 1])
    else:
        cc_slice = 0.0

    if np.std(model_gt) > 1e-12 and np.std(model_rec) > 1e-12:
        cc_vol = float(np.corrcoef(model_gt, model_rec)[0, 1])
    else:
        cc_vol = 0.0
    
    re_vol = float(np.linalg.norm(model_gt - model_rec) /
                   max(np.linalg.norm(model_gt), 1e-12))

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
    
    print("\n[EVAL] Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:25s} = {v:.6f}")
    
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), model_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), model_gt)
    
    try:
        visualize_results(mesh_info, model_gt, model_rec, rx_locs,
                          d_clean, d_noisy, d_rec, metrics,
                          os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")
    
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

    im = axes[0, 0].imshow(gt_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'(a) GT Density (z-slice {iz})')
    plt.colorbar(im, ax=axes[0, 0], label='Δρ [g/cm³]')

    im = axes[0, 1].imshow(rec_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 1].set_title('(b) Inversion Result')
    plt.colorbar(im, ax=axes[0, 1], label='Δρ [g/cm³]')

    err = gt_3d[:, :, iz] - rec_3d[:, :, iz]
    im = axes[0, 2].imshow(err.T, cmap='RdBu_r',
                            vmin=-vmax/2, vmax=vmax/2, origin='lower')
    axes[0, 2].set_title('(c) Error')
    plt.colorbar(im, ax=axes[0, 2], label='Δρ error')

    n_rx = int(np.sqrt(len(d_clean)))
    if n_rx ** 2 == len(d_clean):
        d_map = d_clean.reshape(n_rx, n_rx)
        axes[1, 0].imshow(d_map, cmap='viridis', origin='lower')
    else:
        axes[1, 0].scatter(rx_locs[:, 0], rx_locs[:, 1],
                           c=d_clean, cmap='viridis', s=20)
    axes[1, 0].set_title('(d) Gravity Anomaly (GT)')

    axes[1, 1].plot(d_clean, d_rec, 'b.', ms=3)
    lims = [min(d_clean.min(), d_rec.min()),
            max(d_clean.max(), d_rec.max())]
    axes[1, 1].plot(lims, lims, 'k--', lw=0.5)
    axes[1, 1].set_xlabel('True g_z [mGal]')
    axes[1, 1].set_ylabel('Predicted g_z [mGal]')
    axes[1, 1].set_title(f'(e) Data Fit  CC={metrics["CC_data"]:.4f}')

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


# ============================================================================
# Main Test Logic
# ============================================================================

def main():
    data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Categorize files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Found {len(outer_files)} outer files, {len(inner_files)} inner files")
    
    if not outer_files:
        print("[ERROR] No outer data files found!")
        sys.exit(1)
    
    # Load outer (primary) data
    outer_path = outer_files[0]
    print(f"[INFO] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args/kwargs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Outer data keys: {outer_data.keys()}")
    print(f"[INFO] Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's function
    print("\n[INFO] Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
        print(f"[INFO] Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
    except Exception as e:
        print(f"[ERROR] Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if chained execution
    if inner_files:
        # Chained execution pattern
        print("\n[INFO] Chained execution detected")
        inner_path = inner_files[0]
        print(f"[INFO] Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Agent output should be callable
        if callable(agent_output):
            print("[INFO] Calling agent_output with inner args...")
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Calling agent_output failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("[WARN] Agent output not callable, using directly")
            final_result = agent_output
    else:
        # Direct execution pattern
        print("\n[INFO] Direct execution pattern")
        final_result = agent_output
        std_result = std_output
    
    # For evaluation, we need the data_dict which contains model_gt, etc.
    # The input to run_inversion is data_dict
    if len(args) > 0:
        data_dict = args[0]
    elif 'data_dict' in kwargs:
        data_dict = kwargs['data_dict']
    else:
        print("[ERROR] Cannot find data_dict in inputs")
        sys.exit(1)
    
    # Evaluate agent's result
    print("\n" + "="*60)
    print("[INFO] Evaluating AGENT's reconstruction...")
    print("="*60)
    try:
        metrics_agent = evaluate_results(data_dict, final_result)
    except Exception as e:
        print(f"[ERROR] Evaluation of agent result failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("\n" + "="*60)
    print("[INFO] Evaluating STANDARD reconstruction...")
    print("="*60)
    try:
        metrics_std = evaluate_results(data_dict, std_result)
    except Exception as e:
        print(f"[ERROR] Evaluation of standard result failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare metrics
    print("\n" + "="*60)
    print("[COMPARISON] Agent vs Standard")
    print("="*60)
    
    # Extract key metrics for comparison
    # For this inversion problem:
    # - PSNR: Higher is better
    # - SSIM: Higher is better (closer to 1)
    # - CC (Correlation Coefficient): Higher is better
    # - RE (Relative Error): Lower is better
    # - RMSE: Lower is better
    
    psnr_agent = metrics_agent.get('PSNR_slice', 0)
    psnr_std = metrics_std.get('PSNR_slice', 0)
    
    ssim_agent = metrics_agent.get('SSIM_slice', 0)
    ssim_std = metrics_std.get('SSIM_slice', 0)
    
    cc_vol_agent = metrics_agent.get('CC_volume', 0)
    cc_vol_std = metrics_std.get('CC_volume', 0)
    
    re_vol_agent = metrics_agent.get('RE_volume', 1)
    re_vol_std = metrics_std.get('RE_volume', 1)
    
    print(f"PSNR_slice  -> Agent: {psnr_agent:.4f}, Standard: {psnr_std:.4f}")
    print(f"SSIM_slice  -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
    print(f"CC_volume   -> Agent: {cc_vol_agent:.4f}, Standard: {cc_vol_std:.4f}")
    print(f"RE_volume   -> Agent: {re_vol_agent:.4f}, Standard: {re_vol_std:.4f}")
    
    # Determine success based on multiple criteria
    # We allow some margin of error (10% degradation)
    margin = 0.90  # Allow 10% worse performance
    
    checks_passed = 0
    total_checks = 4
    
    # PSNR check (higher is better)
    if psnr_agent >= psnr_std * margin:
        checks_passed += 1
        print(f"[PASS] PSNR check passed")
    else:
        print(f"[FAIL] PSNR check failed: {psnr_agent:.4f} < {psnr_std * margin:.4f}")
    
    # SSIM check (higher is better)
    if ssim_agent >= ssim_std * margin:
        checks_passed += 1
        print(f"[PASS] SSIM check passed")
    else:
        print(f"[FAIL] SSIM check failed: {ssim_agent:.4f} < {ssim_std * margin:.4f}")
    
    # CC_volume check (higher is better)
    if cc_vol_agent >= cc_vol_std * margin:
        checks_passed += 1
        print(f"[PASS] CC_volume check passed")
    else:
        print(f"[FAIL] CC_volume check failed: {cc_vol_agent:.4f} < {cc_vol_std * margin:.4f}")
    
    # RE_volume check (lower is better)
    # For lower-is-better metrics, agent should be <= std * (1/margin) or std * 1.1
    re_margin = 1.0 / margin  # 1.11, meaning allow up to 11% worse
    if re_vol_agent <= re_vol_std * re_margin:
        checks_passed += 1
        print(f"[PASS] RE_volume check passed")
    else:
        print(f"[FAIL] RE_volume check failed: {re_vol_agent:.4f} > {re_vol_std * re_margin:.4f}")
    
    print(f"\n[SUMMARY] Checks passed: {checks_passed}/{total_checks}")
    
    # Require at least 3 out of 4 checks to pass
    if checks_passed >= 3:
        print("\n[SUCCESS] Agent performance is acceptable!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Agent performance degraded significantly!")
        sys.exit(1)


if __name__ == "__main__":
    main()