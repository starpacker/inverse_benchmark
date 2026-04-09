import matplotlib

matplotlib.use('Agg')

import os

import sys

import json

import logging

import numpy as np

import matplotlib.pyplot as plt

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

sys.path.insert(0, REPO_DIR)

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(result, results_dir=None, save_outputs=True):
    """
    Evaluate the DIC inversion results by computing metrics and creating visualizations.
    
    Parameters
    ----------
    result : dict
        Output from run_inversion containing DIC and ground truth fields.
    results_dir : str or None
        Directory to save outputs. If None, uses RESULTS_DIR.
    save_outputs : bool
        Whether to save output files.
    
    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics:
        - displacement_rmse_ux, displacement_rmse_uy, displacement_rmse_magnitude
        - displacement_psnr_dB, displacement_cc, displacement_ssim
        - strain_rmse, strain_psnr_dB, strain_cc
        - max_gt_displacement_pixels
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    gt_ux = result['gt_ux']
    gt_uy = result['gt_uy']
    dic_ux = result['dic_ux']
    dic_uy = result['dic_uy']
    gt_exx = result['gt_exx']
    gt_eyy = result['gt_eyy']
    gt_exy = result['gt_exy']
    eps_xx = result['eps_xx']
    eps_yy = result['eps_yy']
    eps_xy = result['eps_xy']
    
    # --- Displacement metrics ---
    rmse_ux = float(np.sqrt(np.mean((dic_ux - gt_ux) ** 2)))
    rmse_uy = float(np.sqrt(np.mean((dic_uy - gt_uy) ** 2)))
    disp_mag_gt = np.sqrt(gt_ux**2 + gt_uy**2)
    disp_mag_dic = np.sqrt(dic_ux**2 + dic_uy**2)
    rmse_disp = float(np.sqrt(np.mean((disp_mag_dic - disp_mag_gt)**2)))
    max_disp = float(np.max(np.abs(disp_mag_gt))) + 1e-12
    
    # Displacement PSNR
    mse_disp = float(np.mean((disp_mag_dic - disp_mag_gt)**2))
    psnr_disp = float(10.0 * np.log10(max_disp**2 / (mse_disp + 1e-20)))
    
    # Displacement CC
    a = (disp_mag_dic - disp_mag_dic.mean()).ravel()
    b = (disp_mag_gt - disp_mag_gt.mean()).ravel()
    cc_disp = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 else 0.0
    
    # --- Strain metrics ---
    # Trim borders where finite-diff gradient is unreliable
    s = 2  # border trim
    def _trim(arr):
        return arr[s:-s, s:-s]
    
    strain_gt_flat = np.stack([_trim(gt_exx), _trim(gt_eyy), _trim(gt_exy)]).ravel()
    strain_dic_flat = np.stack([_trim(eps_xx), _trim(eps_yy), _trim(eps_xy)]).ravel()
    
    rmse_strain = float(np.sqrt(np.mean((strain_dic_flat - strain_gt_flat)**2)))
    max_strain = float(np.max(np.abs(strain_gt_flat))) + 1e-12
    mse_strain = float(np.mean((strain_dic_flat - strain_gt_flat)**2))
    psnr_strain = float(10.0 * np.log10(max_strain**2 / (mse_strain + 1e-20)))
    
    cc_strain_a = (strain_dic_flat - strain_dic_flat.mean())
    cc_strain_b = (strain_gt_flat - strain_gt_flat.mean())
    cc_strain = float(np.corrcoef(cc_strain_a, cc_strain_b)[0, 1]) if np.std(cc_strain_a) > 1e-12 else 0.0
    
    # --- SSIM-like metric for displacement field ---
    mu_x = disp_mag_dic.mean()
    mu_y = disp_mag_gt.mean()
    sig_x = disp_mag_dic.std()
    sig_y = disp_mag_gt.std()
    sig_xy = np.mean((disp_mag_dic - mu_x) * (disp_mag_gt - mu_y))
    C1 = (0.01 * max_disp)**2
    C2 = (0.03 * max_disp)**2
    ssim_disp = float(((2*mu_x*mu_y + C1)*(2*sig_xy + C2)) /
                       ((mu_x**2 + mu_y**2 + C1)*(sig_x**2 + sig_y**2 + C2)))
    
    metrics = {
        "displacement_rmse_ux": round(rmse_ux, 6),
        "displacement_rmse_uy": round(rmse_uy, 6),
        "displacement_rmse_magnitude": round(rmse_disp, 6),
        "displacement_psnr_dB": round(psnr_disp, 2),
        "displacement_cc": round(cc_disp, 6),
        "displacement_ssim": round(ssim_disp, 6),
        "strain_rmse": round(rmse_strain, 6),
        "strain_psnr_dB": round(psnr_strain, 2),
        "strain_cc": round(cc_strain, 6),
        "max_gt_displacement_pixels": round(max_disp, 4),
    }
    
    if save_outputs:
        # Save ground truth
        gt_data = {
            'gt_ux': gt_ux,
            'gt_uy': gt_uy,
            'gt_exx': gt_exx,
            'gt_eyy': gt_eyy,
            'gt_exy': gt_exy,
        }
        np.save(os.path.join(results_dir, "ground_truth.npy"), gt_data)
        
        # Save reconstruction
        recon_data = {
            'dic_ux': dic_ux,
            'dic_uy': dic_uy,
            'eps_xx': eps_xx,
            'eps_yy': eps_yy,
            'eps_xy': eps_xy,
        }
        np.save(os.path.join(results_dir, "reconstruction.npy"), recon_data)
        
        # Save metrics
        with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualization
        image_stack = result['image_stack']
        mesh = result['mesh']
        fields = result['fields']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # (a) Reference image with mesh overlay
        ax = axes[0, 0]
        ref_img = np.array(image_stack[0])
        ax.imshow(ref_img, cmap='gray', origin='upper')
        # Draw mesh lines
        mesh_obj = mesh
        for node_x in np.linspace(mesh_obj.Xc1, mesh_obj.Xc2, mesh_obj.n_elx + 1):
            ax.axvline(x=node_x, color='cyan', linewidth=0.5, alpha=0.6)
        for node_y in np.linspace(mesh_obj.Yc1, mesh_obj.Yc2, mesh_obj.n_ely + 1):
            ax.axhline(y=node_y, color='cyan', linewidth=0.5, alpha=0.6)
        ax.set_title("(a) Reference Image + Mesh", fontsize=12)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        
        # (b) Deformed image
        ax = axes[0, 1]
        def_img = np.array(image_stack[1])
        ax.imshow(def_img, cmap='gray', origin='upper')
        ax.set_title("(b) Deformed Image", fontsize=12)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        
        # (c) Displacement magnitude: DIC result
        ax = axes[1, 0]
        im = ax.imshow(disp_mag_dic, cmap='viridis', origin='upper', aspect='equal')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Displacement [px]")
        # Overlay GT contours
        ax.contour(disp_mag_gt, levels=5, colors='red', linewidths=0.8, linestyles='--')
        ax.set_title(f"(c) Disp. Magnitude (DIC)\nRMSE={metrics['displacement_rmse_magnitude']:.4f} px, CC={metrics['displacement_cc']:.4f}",
                     fontsize=11)
        ax.set_xlabel("Element e-coord")
        ax.set_ylabel("Element n-coord")
        
        # (d) Strain error map
        ax = axes[1, 1]
        strain_err = eps_xx[s:-s, s:-s] - gt_exx[s:-s, s:-s]
        vmax = max(abs(strain_err.min()), abs(strain_err.max())) or 1e-6
        im = ax.imshow(strain_err, cmap='RdBu_r', origin='upper', aspect='equal',
                       vmin=-vmax, vmax=vmax)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("εxx error")
        ax.set_title(f"(d) Strain εxx Error (DIC − GT)\nStrain RMSE={metrics['strain_rmse']:.6f}",
                     fontsize=11)
        ax.set_xlabel("Element e-coord")
        ax.set_ylabel("Element n-coord")
        
        plt.suptitle("Task 167: muDIC — Digital Image Correlation\n"
                     f"Displacement PSNR={metrics['displacement_psnr_dB']:.1f} dB, "
                     f"SSIM={metrics['displacement_ssim']:.4f}",
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved reconstruction_result.png")
        print(f"[INFO] Saved ground_truth.npy, reconstruction.npy, metrics.json")
    
    return metrics
