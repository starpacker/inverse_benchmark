import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_101_pyilc_cmb"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(cmb_gt, cmb_rec, data, freqs_ghz, weights, results_dir, assets_dir):
    """
    Compute metrics, save outputs, and generate visualization.
    
    Parameters
    ----------
    cmb_gt : np.ndarray, shape (n_pix, n_pix)
        Ground truth CMB map.
    cmb_rec : np.ndarray, shape (n_pix, n_pix)
        Recovered CMB map.
    data : np.ndarray, shape (n_freq, n_pix, n_pix)
        Multi-frequency observations.
    freqs_ghz : np.ndarray
        Observation frequencies.
    weights : np.ndarray, shape (n_freq,)
        ILC weights.
    results_dir : str
        Path to results directory.
    assets_dir : str
        Path to assets directory.
    
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, SSIM, CC, RMSE.
    """
    # Compute metrics
    mse = np.mean((cmb_gt - cmb_rec)**2)
    data_range = cmb_gt.max() - cmb_gt.min()
    psnr = 10.0 * np.log10(data_range**2 / mse) if mse > 0 else 100.0
    ssim_val = ssim(cmb_gt, cmb_rec, data_range=data_range)
    cc = float(np.corrcoef(cmb_gt.ravel(), cmb_rec.ravel())[0, 1])
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "CC": float(cc),
        "RMSE": float(rmse),
    }
    
    # Save outputs
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), cmb_gt)
        np.save(os.path.join(d, "recon_output.npy"), cmb_rec)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    n_pix = cmb_gt.shape[0]
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: selected frequency maps (3 of 6)
    sel = [0, 2, 5]  # 30, 70, 217 GHz
    for idx, si in enumerate(sel):
        ax = fig.add_subplot(3, 3, idx + 1)
        vmax = np.percentile(np.abs(data[si]), 99)
        ax.imshow(data[si], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"{freqs_ghz[si]:.0f} GHz", fontsize=10)
        ax.axis('off')
    
    # Row 2: GT CMB, recovered CMB, residual
    vmax_cmb = np.percentile(np.abs(cmb_gt), 99)
    
    ax = fig.add_subplot(3, 3, 4)
    ax.imshow(cmb_gt, cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title("GT CMB", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(cmb_rec, cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title("Recovered CMB (ILC)", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(3, 3, 6)
    residual = cmb_gt - cmb_rec
    vmax_res = np.percentile(np.abs(residual), 99)
    ax.imshow(residual, cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res)
    ax.set_title(f"Residual (RMS={np.std(residual):.1f} μK)", fontsize=10)
    ax.axis('off')
    
    # Row 3 left: ILC weights
    ax = fig.add_subplot(3, 3, 7)
    ax.bar(range(len(freqs_ghz)), weights, color='steelblue')
    ax.set_xticks(range(len(freqs_ghz)))
    ax.set_xticklabels([f"{f:.0f}" for f in freqs_ghz], fontsize=8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Weight")
    ax.set_title("ILC Weights")
    ax.axhline(0, color='k', lw=0.5)
    
    # Row 3 middle: power spectra comparison
    ax = fig.add_subplot(3, 3, 8)
    ps_gt = np.abs(np.fft.fft2(cmb_gt))**2
    ps_rec = np.abs(np.fft.fft2(cmb_rec))**2
    k = np.arange(1, n_pix // 2)
    kx = np.fft.fftfreq(n_pix, d=1.0) * n_pix
    ky = np.fft.fftfreq(n_pix, d=1.0) * n_pix
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    cl_gt = np.zeros(len(k))
    cl_rec = np.zeros(len(k))
    for i, ki in enumerate(k):
        mask = (K >= ki - 0.5) & (K < ki + 0.5)
        if mask.sum() > 0:
            cl_gt[i] = ps_gt[mask].mean()
            cl_rec[i] = ps_rec[mask].mean()
    
    ax.loglog(k, cl_gt, 'b-', label='GT', lw=1.5)
    ax.loglog(k, cl_rec, 'r--', label='ILC', lw=1.5)
    ax.set_xlabel("Multipole ℓ")
    ax.set_ylabel("C_ℓ")
    ax.set_title("Angular Power Spectrum")
    ax.legend(fontsize=8)
    
    # Row 3 right: metrics text
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    txt = (f"PSNR = {metrics['PSNR']:.2f} dB\n"
           f"SSIM = {metrics['SSIM']:.4f}\n"
           f"CC   = {metrics['CC']:.4f}\n"
           f"RMSE = {metrics['RMSE']:.2f} μK\n"
           f"\nΣ weights = {weights.sum():.6f}")
    ax.text(0.1, 0.5, txt, fontsize=14, family='monospace',
            transform=ax.transAxes, verticalalignment='center')
    ax.set_title("Metrics Summary")
    
    plt.tight_layout()
    for path in [os.path.join(results_dir, "vis_result.png"),
                 os.path.join(assets_dir, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return metrics
