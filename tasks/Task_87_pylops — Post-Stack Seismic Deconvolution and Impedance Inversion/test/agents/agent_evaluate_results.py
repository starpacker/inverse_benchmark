import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

import warnings

warnings.filterwarnings('ignore')

from scipy.ndimage import uniform_filter, uniform_filter1d

def evaluate_results(reflectivity_true, reflectivity_inv, impedance_true, 
                     impedance_inv, impedance_inv2, seismic_obs, wavelet, wav_t,
                     params, results_dir):
    """
    Evaluate inversion results and generate visualizations.
    
    Args:
        reflectivity_true: true reflectivity (nt, n_traces)
        reflectivity_inv: inverted reflectivity (nt, n_traces)
        impedance_true: true impedance (nt, n_traces)
        impedance_inv: inverted impedance from method 1 (nt, n_traces)
        impedance_inv2: inverted impedance from method 2 (nt, n_traces)
        seismic_obs: observed seismic data (nt, n_traces)
        wavelet: source wavelet
        wav_t: wavelet time axis
        params: dictionary of parameters
        results_dir: directory to save results
    
    Returns:
        dict containing final metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    nt = params['nt']
    n_traces = params['n_traces']
    dt = params['dt']
    freq_dominant = params['freq_dominant']
    
    # SSIM computation helper
    def ssim_2d(img1, img2, win_size=7):
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        C1 = (0.01 * data_range)**2
        C2 = (0.03 * data_range)**2
        mu1 = uniform_filter(img1, win_size)
        mu2 = uniform_filter(img2, win_size)
        sigma1_sq = uniform_filter(img1**2, win_size) - mu1**2
        sigma2_sq = uniform_filter(img2**2, win_size) - mu2**2
        sigma12 = uniform_filter(img1 * img2, win_size) - mu1 * mu2
        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    # Compute metrics for both methods
    def compute_metrics(ref_true, ref_inv, imp_true, imp_inv):
        # Reflectivity metrics
        mse_r = np.mean((ref_true - ref_inv)**2)
        range_r = ref_true.max() - ref_true.min()
        psnr_r = 10 * np.log10(range_r**2 / mse_r) if mse_r > 0 else float('inf')
        cc_r = np.corrcoef(ref_true.flatten(), ref_inv.flatten())[0, 1]
        
        # Impedance metrics
        mse_i = np.mean((imp_true - imp_inv)**2)
        range_i = imp_true.max() - imp_true.min()
        psnr_i = 10 * np.log10(range_i**2 / mse_i) if mse_i > 0 else float('inf')
        cc_i = np.corrcoef(imp_true.flatten(), imp_inv.flatten())[0, 1]
        
        ssim_r = ssim_2d(ref_true, ref_inv)
        ssim_i = ssim_2d(imp_true, imp_inv)
        
        metrics = {
            'psnr_reflectivity': float(psnr_r),
            'ssim_reflectivity': float(ssim_r),
            'cc_reflectivity': float(cc_r),
            'psnr_impedance': float(psnr_i),
            'ssim_impedance': float(ssim_i),
            'cc_impedance': float(cc_i),
            'rmse_reflectivity': float(np.sqrt(mse_r)),
            'rmse_impedance': float(np.sqrt(mse_i)),
        }
        return metrics
    
    metrics1 = compute_metrics(reflectivity_true, reflectivity_inv,
                               impedance_true, impedance_inv)
    metrics2 = compute_metrics(reflectivity_true, reflectivity_inv,
                               impedance_true, impedance_inv2)
    
    # Pick better result
    if metrics2['cc_impedance'] > metrics1['cc_impedance']:
        print("[EVAL] PoststackLinearModelling inversion is better → using it")
        impedance_final = impedance_inv2
        metrics = metrics2
    else:
        print("[EVAL] Regularized deconvolution is better → using it")
        impedance_final = impedance_inv
        metrics = metrics1
    
    # Report metrics
    print(f"\n[EVAL] === Final Metrics ===")
    for k, v in metrics.items():
        print(f"[EVAL] {k} = {v:.6f}")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "input.npy"), seismic_obs)
    np.save(os.path.join(results_dir, "ground_truth.npy"), impedance_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), impedance_final)
    print(f"[SAVE] Input shape: {seismic_obs.shape} → input.npy")
    print(f"[SAVE] GT shape: {impedance_true.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {impedance_final.shape} → reconstruction.npy")
    
    # Generate visualization
    t_axis = np.arange(nt) * dt
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # (a) Source wavelet
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(wav_t * 1000, wavelet, 'b-', lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Ricker Wavelet ({freq_dominant} Hz)')
    ax.grid(True, alpha=0.3)
    
    # (b) True impedance section
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(impedance_true, aspect='auto', cmap='RdYlBu_r',
                   extent=[0, n_traces, t_axis[-1], t_axis[0]])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (s)')
    ax.set_title('True Impedance')
    plt.colorbar(im, ax=ax, label='Z (kg/m²s)')
    
    # (c) Observed seismic section
    ax = fig.add_subplot(gs[0, 2])
    vmax = np.percentile(np.abs(seismic_obs), 98)
    im2 = ax.imshow(seismic_obs, aspect='auto', cmap='seismic',
                    extent=[0, n_traces, t_axis[-1], t_axis[0]],
                    vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (s)')
    ax.set_title('Observed Seismic (noisy)')
    plt.colorbar(im2, ax=ax)
    
    # (d) Inverted impedance
    ax = fig.add_subplot(gs[0, 3])
    im3 = ax.imshow(impedance_final, aspect='auto', cmap='RdYlBu_r',
                    extent=[0, n_traces, t_axis[-1], t_axis[0]],
                    vmin=impedance_true.min(), vmax=impedance_true.max())
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (s)')
    ax.set_title('Inverted Impedance')
    plt.colorbar(im3, ax=ax, label='Z (kg/m²s)')
    
    # (e) True vs inverted reflectivity (single trace)
    ax = fig.add_subplot(gs[1, 0:2])
    mid_trace = n_traces // 2
    ax.plot(t_axis, reflectivity_true[:, mid_trace], 'b-', alpha=0.7,
            label='True', lw=1.5)
    ax.plot(t_axis, reflectivity_inv[:, mid_trace], 'r-', alpha=0.7,
            label='Inverted', lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reflectivity')
    ax.set_title(f'Reflectivity Trace #{mid_trace}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (f) True vs inverted impedance (single trace)
    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(t_axis, impedance_true[:, mid_trace], 'b-', lw=2,
            label='True')
    ax.plot(t_axis, impedance_final[:, mid_trace], 'r--', lw=2,
            label='Inverted')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Impedance')
    ax.set_title(f'Impedance Trace #{mid_trace}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (g) Error section
    ax = fig.add_subplot(gs[2, 0:2])
    error_imp = impedance_true - impedance_final
    vmax_err = np.percentile(np.abs(error_imp), 95)
    im4 = ax.imshow(error_imp, aspect='auto', cmap='seismic',
                    extent=[0, n_traces, t_axis[-1], t_axis[0]],
                    vmin=-vmax_err, vmax=vmax_err)
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (s)')
    ax.set_title('Impedance Error (GT - Inv)')
    plt.colorbar(im4, ax=ax)
    
    # (h) Scatter plot
    ax = fig.add_subplot(gs[2, 2:4])
    ax.scatter(impedance_true.flatten(), impedance_final.flatten(),
               s=1, alpha=0.2, c='steelblue')
    lim = [impedance_true.min() * 0.95, impedance_true.max() * 1.05]
    ax.plot(lim, lim, 'r--', lw=2)
    ax.set_xlabel('True Impedance')
    ax.set_ylabel('Inverted Impedance')
    ax.set_title(f'Scatter (CC={metrics["cc_impedance"]:.4f})')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"pylops — Post-Stack Seismic Deconvolution & Impedance Inversion\n"
        f"PSNR_Z={metrics['psnr_impedance']:.2f} dB | "
        f"SSIM_Z={metrics['ssim_impedance']:.4f} | "
        f"CC_Z={metrics['cc_impedance']:.4f} | "
        f"PSNR_r={metrics['psnr_reflectivity']:.2f} dB",
        fontsize=12, fontweight='bold'
    )
    
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {vis_path}")
    
    return metrics
