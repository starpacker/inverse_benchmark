import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

import warnings

warnings.filterwarnings('ignore')

def build_step(depth_list, values):
    """Build step-function arrays for plotting."""
    d_plot = [0]
    v_plot = [values[0]]
    for i, d in enumerate(depth_list):
        d_plot.append(d)
        v_plot.append(values[i])
        d_plot.append(d)
        v_plot.append(values[i+1] if i+1 < len(values) else values[-1])
    d_plot.append(depth_list[-1] + 500)
    v_plot.append(values[-1])
    return d_plot, v_plot

def evaluate_results(data_dict, result_dict, results_dir):
    """
    Evaluate inversion quality, compute metrics, save results, and visualize.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data
        result_dict: Dictionary from run_inversion
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    res_true = data_dict['res_true']
    res_inv = result_dict['res_inv']
    param_indices = data_dict['param_indices']
    data_true = data_dict['data_true']
    data_inv = result_dict['data_inv']
    data_obs = data_dict['data_obs']
    depth = data_dict['depth']
    frequencies = data_dict['frequencies']
    offsets = data_dict['offsets']
    errors = result_dict['errors']
    water_depth = data_dict['water_depth']
    
    # Parameter recovery
    true_params = np.array([res_true[i] for i in param_indices])
    inv_params = np.array([res_inv[i] for i in param_indices])
    
    # Log-space metrics (resistivity spans orders of magnitude)
    log_true = np.log10(true_params)
    log_inv = np.log10(inv_params)
    
    param_rmse = np.sqrt(np.mean((log_true - log_inv)**2))
    param_cc = np.corrcoef(log_true, log_inv)[0, 1]
    
    # Relative errors per layer
    rel_errors = np.abs(inv_params - true_params) / true_params
    
    # Data fit metrics
    d_true = np.abs(data_true.flatten())
    d_inv = np.abs(data_inv.flatten())
    data_cc = np.corrcoef(d_true, d_inv)[0, 1]
    
    # PSNR on log-amplitude
    log_amp_true = np.log10(np.abs(data_true.flatten()) + 1e-20)
    log_amp_inv = np.log10(np.abs(data_inv.flatten()) + 1e-20)
    mse_data = np.mean((log_amp_true - log_amp_inv)**2)
    range_data = log_amp_true.max() - log_amp_true.min()
    psnr_data = 10 * np.log10(range_data**2 / mse_data) if mse_data > 0 else float('inf')
    
    # Reservoir is the middle subsurface layer (index 1 in param_indices)
    reservoir_idx = 1  # Index in true_params/inv_params array
    
    metrics = {
        'param_rmse_log10': float(param_rmse),
        'param_cc': float(param_cc),
        'data_fit_cc': float(data_cc),
        'psnr_data_log': float(psnr_data),
        'reservoir_res_true': float(true_params[reservoir_idx]),
        'reservoir_res_inv': float(inv_params[reservoir_idx]),
        'reservoir_rel_error': float(rel_errors[reservoir_idx]),
        'mean_rel_error': float(np.mean(rel_errors)),
        'max_rel_error': float(np.max(rel_errors)),
    }
    
    # Print metrics
    for k, v in metrics.items():
        print(f"[EVAL] {k} = {v:.6f}")
    
    print(f"\n[EVAL] Layer-by-layer recovery:")
    layer_names_print = [f'Layer {i+1}' for i in range(len(true_params))]
    for i, (name, t, inv, err) in enumerate(zip(layer_names_print, true_params, inv_params, rel_errors)):
        print(f"  {name}: true={t:.2f}, inv={inv:.2f}, error={err*100:.1f}%")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "input.npy"), data_obs)
    np.save(os.path.join(results_dir, "ground_truth.npy"), np.array(res_true))
    np.save(os.path.join(results_dir, "reconstruction.npy"), np.array(res_inv))
    
    # Visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # (a) Resistivity model comparison
    ax = axes[0, 0]
    d_true_plot, v_true_plot = build_step(depth, res_true)
    d_inv_plot, v_inv_plot = build_step(depth, res_inv)
    
    ax.semilogx(v_true_plot, d_true_plot, 'b-', lw=2, label='True')
    ax.semilogx(v_inv_plot, d_inv_plot, 'r--', lw=2, label='Inverted')
    ax.axhline(y=water_depth, color='cyan', ls=':', lw=1, label='Seafloor')
    if len(depth) >= 4:
        ax.axhspan(depth[2], depth[3], alpha=0.2, color='gold', label='Reservoir')
    ax.set_xlabel('Resistivity (Ωm)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('1D Resistivity Model')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 200])
    
    # (b) E-field amplitude vs offset (selected frequencies)
    ax = axes[0, 1]
    freq_indices = [0, 5, 10, 14]
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_indices)))
    for i, fi in enumerate(freq_indices):
        amp_true = np.abs(data_true[fi, :])
        amp_obs = np.abs(data_obs[fi, :])
        amp_inv = np.abs(data_inv[fi, :])
        ax.semilogy(offsets/1000, amp_true, '-', color=colors[i], lw=2,
                    label=f'{frequencies[fi]:.2f} Hz')
        ax.semilogy(offsets/1000, amp_obs, 'o', color=colors[i], ms=6, alpha=0.5)
        ax.semilogy(offsets/1000, amp_inv, 's', color=colors[i], ms=4, alpha=0.8)
    ax.set_xlabel('Offset (km)')
    ax.set_ylabel('|E| (V/Am²)')
    ax.set_title('E-field Amplitude vs Offset')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # (c) Phase vs frequency (selected offsets)
    ax = axes[0, 2]
    off_indices = [0, 2, 4, 6]
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(off_indices)))
    for i, oi in enumerate(off_indices):
        phase_true = np.angle(data_true[:, oi], deg=True)
        phase_inv = np.angle(data_inv[:, oi], deg=True)
        ax.semilogx(frequencies, phase_true, '-', color=colors2[i], lw=2,
                    label=f'{offsets[oi]/1000:.0f} km')
        ax.semilogx(frequencies, phase_inv, 's', color=colors2[i], ms=4, alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (°)')
    ax.set_title('Phase vs Frequency')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # (d) Relative error per layer
    ax = axes[1, 0]
    layer_names = [f'Layer {i+1}' for i in range(len(rel_errors))]
    colors_bar = ['gray'] * len(rel_errors)
    if len(rel_errors) >= 2:
        colors_bar[1] = 'gold'  # Reservoir is the 2nd subsurface layer
    bars = ax.bar(range(len(rel_errors)), rel_errors * 100, color=colors_bar, edgecolor='black')
    ax.set_xticks(range(len(rel_errors)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Layer Resistivity Recovery')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rel_errors):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # (e) Convergence
    ax = axes[1, 1]
    if len(errors) > 0:
        ax.semilogy(range(1, len(errors)+1), errors, 'b-o', lw=2, ms=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weighted RMS Misfit')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)
    
    # (f) True vs Inverted parameters (log scale)
    ax = axes[1, 2]
    ax.scatter(np.log10(true_params), np.log10(inv_params), c=colors_bar,
               s=100, edgecolors='black', zorder=5)
    lim = [min(np.log10(true_params).min(), np.log10(inv_params).min()) - 0.2,
           max(np.log10(true_params).max(), np.log10(inv_params).max()) + 0.2]
    ax.plot(lim, lim, 'r--', lw=2)
    for i, name in enumerate(layer_names):
        ax.annotate(name, (np.log10(true_params[i]), np.log10(inv_params[i])),
                   fontsize=7, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('log₁₀(True ρ)')
    ax.set_ylabel('log₁₀(Inverted ρ)')
    ax.set_title(f'Parameter Recovery (CC={metrics["param_cc"]:.4f})')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"empymod — 1D CSEM Resistivity Inversion (Differential Evolution)\n"
        f"Param CC={metrics['param_cc']:.4f} | Data PSNR={metrics['psnr_data_log']:.2f} dB | "
        f"Reservoir error={metrics['reservoir_rel_error']*100:.1f}%",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {vis_path}")
    
    return metrics
