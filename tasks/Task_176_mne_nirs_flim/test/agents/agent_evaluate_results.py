import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import os

import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(hbo_gt, hbr_gt, hbo_rec, hbr_rec, t, od_760_noisy, od_850_noisy,
                     block_starts, block_duration, save_plots=True, save_metrics=True):
    """
    Compute metrics for both HbO and HbR, create visualizations, and save results.
    
    Parameters
    ----------
    hbo_gt : ndarray
        Ground truth HbO concentration
    hbr_gt : ndarray
        Ground truth HbR concentration
    hbo_rec : ndarray
        Recovered HbO concentration
    hbr_rec : ndarray
        Recovered HbR concentration
    t : ndarray
        Time vector
    od_760_noisy : ndarray
        Noisy optical density at 760nm
    od_850_noisy : ndarray
        Noisy optical density at 850nm
    block_starts : list
        Start times of stimulus blocks
    block_duration : float
        Duration of each stimulus block
    save_plots : bool
        Whether to save plots
    save_metrics : bool
        Whether to save metrics to JSON
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Compute PSNR
    def compute_psnr(gt, rec):
        mse = np.mean((gt - rec) ** 2)
        if mse < 1e-30:
            return 100.0
        peak = np.max(np.abs(gt))
        return 10 * np.log10(peak ** 2 / mse)
    
    # Compute correlation coefficient
    def compute_cc(gt, rec):
        return float(np.corrcoef(gt, rec)[0, 1])
    
    # Compute RMSE
    def compute_rmse(gt, rec):
        return float(np.sqrt(np.mean((gt - rec) ** 2)))
    
    metrics = {
        'HbO_PSNR_dB': compute_psnr(hbo_gt, hbo_rec),
        'HbO_CC': compute_cc(hbo_gt, hbo_rec),
        'HbO_RMSE': compute_rmse(hbo_gt, hbo_rec),
        'HbR_PSNR_dB': compute_psnr(hbr_gt, hbr_rec),
        'HbR_CC': compute_cc(hbr_gt, hbr_rec),
        'HbR_RMSE': compute_rmse(hbr_gt, hbr_rec),
    }
    
    # Overall averages
    metrics['PSNR_dB'] = (metrics['HbO_PSNR_dB'] + metrics['HbR_PSNR_dB']) / 2
    metrics['CC'] = (metrics['HbO_CC'] + metrics['HbR_CC']) / 2
    metrics['RMSE'] = (metrics['HbO_RMSE'] + metrics['HbR_RMSE']) / 2
    
    # Print metrics
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save metrics
    if save_metrics:
        metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved to {metrics_path}")
    
    # Create visualization
    if save_plots:
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        
        # Helper: shade stimulus blocks
        def shade_blocks(ax):
            for s in block_starts:
                ax.axvspan(s, s + block_duration, color='yellow', alpha=0.2, label=None)
        
        # Panel 1: Ground truth HbO & HbR
        ax = axes[0]
        shade_blocks(ax)
        ax.plot(t, hbo_gt * 1e6, 'r-', lw=1.5, label='HbO (GT)')
        ax.plot(t, hbr_gt * 1e6, 'b-', lw=1.5, label='HbR (GT)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title('Ground Truth Hemodynamic Response')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Noisy optical density input
        ax = axes[1]
        shade_blocks(ax)
        ax.plot(t, od_760_noisy, 'purple', lw=0.8, alpha=0.7, label='ΔOD 760nm')
        ax.plot(t, od_850_noisy, 'orange', lw=0.8, alpha=0.7, label='ΔOD 850nm')
        ax.set_ylabel('Optical Density Change')
        ax.set_title('Noisy Optical Density Input (MBLL Forward + Noise)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Recovered vs GT
        ax = axes[2]
        shade_blocks(ax)
        ax.plot(t, hbo_gt * 1e6, 'r--', lw=1.5, alpha=0.6, label='HbO (GT)')
        ax.plot(t, hbo_rec * 1e6, 'r-', lw=1.2, label='HbO (Recovered)')
        ax.plot(t, hbr_gt * 1e6, 'b--', lw=1.5, alpha=0.6, label='HbR (GT)')
        ax.plot(t, hbr_rec * 1e6, 'b-', lw=1.2, label='HbR (Recovered)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title(f'Recovered Hemodynamic Response  |  PSNR={metrics["PSNR_dB"]:.1f} dB, CC={metrics["CC"]:.4f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Residuals
        ax = axes[3]
        shade_blocks(ax)
        ax.plot(t, (hbo_gt - hbo_rec) * 1e6, 'r-', lw=1.0, label='HbO residual')
        ax.plot(t, (hbr_gt - hbr_rec) * 1e6, 'b-', lw=1.0, label='HbR residual')
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.set_ylabel('Residual (µM)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Residuals (GT − Recovered)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure saved to {fig_path}")
    
    # Save arrays
    gt_stack = np.stack([hbo_gt, hbr_gt], axis=0)
    rec_stack = np.stack([hbo_rec, hbr_rec], axis=0)
    input_stack = np.stack([od_760_noisy, od_850_noisy], axis=0)
    
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), gt_stack)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), rec_stack)
    np.save(os.path.join(RESULTS_DIR, 'input_data.npy'), input_stack)
    print(f"  Arrays saved to {RESULTS_DIR}")
    
    return metrics
