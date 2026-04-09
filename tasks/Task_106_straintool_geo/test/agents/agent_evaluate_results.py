import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_106_straintool_geo"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(data, inversion_result):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, CC metrics for each strain component,
    generates visualizations, and saves outputs.
    
    Args:
        data: dict from load_and_preprocess_data
        inversion_result: dict from run_inversion
        
    Returns:
        dict containing metrics
    """
    gt_exx = data['gt_exx']
    gt_exy = data['gt_exy']
    gt_eyy = data['gt_eyy']
    grid_x = data['grid_x']
    grid_y = data['grid_y']
    stations = data['stations']
    vx = data['vx']
    vy = data['vy']
    
    rec_exx = inversion_result['rec_exx']
    rec_exy = inversion_result['rec_exy']
    rec_eyy = inversion_result['rec_eyy']
    
    def compute_field_metrics(gt, rec):
        """Compute PSNR, SSIM, CC for 2D field comparison."""
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-15)
        rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-15)
        
        # PSNR
        mse = np.mean((gt_n - rec_n)**2)
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
        
        # SSIM
        data_range = max(gt_n.max() - gt_n.min(), rec_n.max() - rec_n.min())
        if data_range < 1e-15:
            data_range = 1.0
        ssim_val = ssim(gt_n, rec_n, data_range=data_range)
        
        # CC
        gt_z = gt_n - gt_n.mean()
        rec_z = rec_n - rec_n.mean()
        denom = np.sqrt(np.sum(gt_z**2) * np.sum(rec_z**2))
        cc = np.sum(gt_z * rec_z) / denom if denom > 1e-15 else 0.0
        
        return float(psnr), float(ssim_val), float(cc)
    
    # Compute metrics per component
    comp_metrics = {}
    all_psnr, all_ssim, all_cc = [], [], []
    
    for name, gt, rec in [("εxx", gt_exx, rec_exx),
                          ("εxy", gt_exy, rec_exy),
                          ("εyy", gt_eyy, rec_eyy)]:
        p, s, c = compute_field_metrics(gt, rec)
        comp_metrics[name] = {"PSNR": p, "SSIM": s, "CC": c}
        all_psnr.append(p)
        all_ssim.append(s)
        all_cc.append(c)
        print(f"    {name}: PSNR={p:.2f}, SSIM={s:.4f}, CC={c:.4f}")
    
    avg_psnr = float(np.mean(all_psnr))
    avg_ssim = float(np.mean(all_ssim))
    avg_cc = float(np.mean(all_cc))
    print(f"\n    Average: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, CC={avg_cc:.4f}")
    
    metrics = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "CC": avg_cc,
        "components": comp_metrics,
    }
    
    # Save outputs
    gt_all = np.stack([gt_exx, gt_exy, gt_eyy])
    rec_all = np.stack([rec_exx, rec_exy, rec_eyy])
    
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_all)
        np.save(os.path.join(d, "recon_output.npy"), rec_all)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    
    components = [
        ("εxx", gt_exx, rec_exx),
        ("εxy", gt_exy, rec_exy),
        ("εyy", gt_eyy, rec_eyy),
    ]
    
    for row, (name, gt, rec) in enumerate(components):
        vmin = min(gt.min(), rec.min())
        vmax = max(gt.max(), rec.max())
        
        ax = axes[row, 0]
        im = ax.imshow(gt, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.scatter(stations[:, 0], stations[:, 1], c='k', s=10, marker='^')
        ax.set_title(f"GT {name} (nanostrain/yr)", fontsize=12)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax)
        
        ax = axes[row, 1]
        im = ax.imshow(rec, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.scatter(stations[:, 0], stations[:, 1], c='k', s=10, marker='^')
        m_comp = comp_metrics[f'{name}']
        ax.set_title(f"Reconstructed {name}\nPSNR={m_comp['PSNR']:.1f}dB, "
                     f"SSIM={m_comp['SSIM']:.3f}, CC={m_comp['CC']:.3f}", fontsize=11)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics
