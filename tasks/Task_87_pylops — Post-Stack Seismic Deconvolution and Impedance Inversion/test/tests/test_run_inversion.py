import sys
import os
import dill
import numpy as np
import traceback
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, uniform_filter1d

# Import the target function
from agent_run_inversion import run_inversion

# Inject the referee evaluation function
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


def compute_score_from_result(result_dict, reflectivity_true, impedance_true, 
                               seismic_obs, wavelet, wav_t, params, results_dir):
    """
    Helper function to compute evaluation score from run_inversion result.
    """
    reflectivity_inv = result_dict['reflectivity_inv']
    impedance_inv = result_dict['impedance_inv']
    impedance_inv2 = result_dict['impedance_inv2']
    
    metrics = evaluate_results(
        reflectivity_true=reflectivity_true,
        reflectivity_inv=reflectivity_inv,
        impedance_true=impedance_true,
        impedance_inv=impedance_inv,
        impedance_inv2=impedance_inv2,
        seismic_obs=seismic_obs,
        wavelet=wavelet,
        wav_t=wav_t,
        params=params,
        results_dir=results_dir
    )
    
    return metrics


def main():
    warnings.filterwarnings('ignore')
    
    # Data paths
    data_paths = ['/data/yjh/pylops_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    if not outer_files:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    # Load outer (primary) data
    outer_path = outer_files[0]
    print(f"\n[LOAD] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[INFO] Outer data keys: {outer_data.keys()}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Args count: {len(args)}")
    print(f"[INFO] Kwargs keys: {kwargs.keys()}")
    
    # Extract inputs for evaluation
    # Based on gen_data_code: run_inversion(seismic_obs, wavelet, impedance_bg, impedance_true)
    seismic_obs = args[0] if len(args) > 0 else kwargs.get('seismic_obs')
    wavelet = args[1] if len(args) > 1 else kwargs.get('wavelet')
    impedance_bg = args[2] if len(args) > 2 else kwargs.get('impedance_bg')
    impedance_true = args[3] if len(args) > 3 else kwargs.get('impedance_true')
    
    print(f"[INFO] seismic_obs shape: {seismic_obs.shape}")
    print(f"[INFO] wavelet shape: {wavelet.shape}")
    print(f"[INFO] impedance_bg shape: {impedance_bg.shape}")
    print(f"[INFO] impedance_true shape: {impedance_true.shape}")
    
    # Construct reflectivity_true from impedance_true
    # r = (Z[i+1] - Z[i]) / (Z[i+1] + Z[i])
    reflectivity_true = np.zeros_like(impedance_true)
    reflectivity_true[1:, :] = (impedance_true[1:, :] - impedance_true[:-1, :]) / \
                               (impedance_true[1:, :] + impedance_true[:-1, :] + 1e-10)
    
    # Construct params for evaluation
    nt, n_traces = seismic_obs.shape
    params = {
        'nt': nt,
        'n_traces': n_traces,
        'dt': 0.004,  # typical seismic sampling rate
        'freq_dominant': 25,  # typical Ricker wavelet frequency
    }
    
    # Construct wav_t (wavelet time axis)
    wav_len = len(wavelet)
    wav_t = np.linspace(-wav_len//2, wav_len//2, wav_len) * params['dt']
    
    # Check for chained execution
    is_chained = len(inner_files) > 0
    
    if is_chained:
        print("\n[MODE] Chained execution detected")
        # Run outer function to get operator
        try:
            print("[RUN] Executing run_inversion to get operator...")
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to run outer function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Load inner data
        inner_path = inner_files[0]
        print(f"\n[LOAD] Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute operator with inner data
        try:
            print("[RUN] Executing operator with inner data...")
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to run operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n[MODE] Direct execution")
        # Run function directly
        try:
            print("[RUN] Executing run_inversion...")
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to run function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        std_result = std_output
    
    print(f"\n[INFO] Agent result keys: {agent_result.keys() if isinstance(agent_result, dict) else type(agent_result)}")
    print(f"[INFO] Std result keys: {std_result.keys() if isinstance(std_result, dict) else type(std_result)}")
    
    # Create results directories
    agent_results_dir = './results_agent'
    std_results_dir = './results_std'
    
    # Evaluate agent result
    print("\n" + "="*60)
    print("[EVAL] Evaluating AGENT result...")
    print("="*60)
    
    try:
        agent_metrics = compute_score_from_result(
            result_dict=agent_result,
            reflectivity_true=reflectivity_true,
            impedance_true=impedance_true,
            seismic_obs=seismic_obs,
            wavelet=wavelet,
            wav_t=wav_t,
            params=params,
            results_dir=agent_results_dir
        )
    except Exception as e:
        print(f"[ERROR] Failed to evaluate agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("\n" + "="*60)
    print("[EVAL] Evaluating STANDARD result...")
    print("="*60)
    
    try:
        std_metrics = compute_score_from_result(
            result_dict=std_result,
            reflectivity_true=reflectivity_true,
            impedance_true=impedance_true,
            seismic_obs=seismic_obs,
            wavelet=wavelet,
            wav_t=wav_t,
            params=params,
            results_dir=std_results_dir
        )
    except Exception as e:
        print(f"[ERROR] Failed to evaluate standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare results
    print("\n" + "="*60)
    print("[COMPARE] Comparing Agent vs Standard")
    print("="*60)
    
    # Use cc_impedance as primary metric (correlation coefficient, higher is better)
    score_agent = agent_metrics['cc_impedance']
    score_std = std_metrics['cc_impedance']
    
    print(f"\nPrimary Metric (CC Impedance):")
    print(f"  Agent:    {score_agent:.6f}")
    print(f"  Standard: {score_std:.6f}")
    
    print(f"\nAll Metrics Comparison:")
    for key in agent_metrics:
        agent_val = agent_metrics[key]
        std_val = std_metrics[key]
        diff_pct = ((agent_val - std_val) / (abs(std_val) + 1e-10)) * 100
        status = "✓" if agent_val >= std_val * 0.9 else "✗"
        print(f"  {key}: Agent={agent_val:.6f}, Std={std_val:.6f}, Diff={diff_pct:+.2f}% {status}")
    
    # Determine success
    # For correlation coefficient: higher is better
    # Allow 10% margin
    threshold = score_std * 0.9
    
    print(f"\n[DECISION] Threshold (90% of standard): {threshold:.6f}")
    print(f"Scores -> Agent: {score_agent:.6f}, Standard: {score_std:.6f}")
    
    if score_agent >= threshold:
        print("\n[RESULT] ✓ PASS - Agent performance is acceptable")
        sys.exit(0)
    else:
        print("\n[RESULT] ✗ FAIL - Agent performance degraded significantly")
        print(f"  Agent CC: {score_agent:.6f} < Threshold: {threshold:.6f}")
        sys.exit(1)


if __name__ == '__main__':
    main()