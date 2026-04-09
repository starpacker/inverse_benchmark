import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import json

import os

from koma.modal import xmacmat, maxreal

def evaluate_results(freq_true, freq_id, zeta_true, zeta_id, phi_true, phi_id,
                     t, acc, results_dir):
    """
    Compute evaluation metrics and create visualization.
    """
    n_dof = len(freq_true)

    # Frequency relative errors
    freq_re = np.abs(freq_id - freq_true) / freq_true

    # Damping ratio relative errors
    zeta_re = np.abs(zeta_id - zeta_true) / zeta_true

    # MAC values
    mac_matrix = xmacmat(phi_true, phi_id, conjugates=False)
    mac_diag = np.diag(mac_matrix)

    metrics = {
        "freq_true": freq_true.tolist(),
        "freq_identified": freq_id.tolist(),
        "freq_relative_errors": freq_re.tolist(),
        "damping_true": zeta_true.tolist(),
        "damping_identified": zeta_id.tolist(),
        "damping_relative_errors": zeta_re.tolist(),
        "mac_values": mac_diag.tolist(),
        "mac_matrix": mac_matrix.tolist(),
        "mean_freq_re": float(np.mean(freq_re)),
        "mean_damping_re": float(np.mean(zeta_re)),
        "mean_mac": float(np.mean(mac_diag)),
        "psnr": None,
        "ssim": None
    }

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Save ground truth and reconstruction
    gt_data = {
        'freq_true': freq_true,
        'zeta_true': zeta_true,
        'phi_true': phi_true,
        'acc_data': acc
    }
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_data, allow_pickle=True)

    recon_data = {
        'freq_id': freq_id,
        'zeta_id': zeta_id,
        'phi_id': phi_id
    }
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_data, allow_pickle=True)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Time series
    ax1 = axes[0, 0]
    t_plot = t[:min(len(t), 2000)]
    for ch in range(n_dof):
        ax1.plot(t_plot, acc[:len(t_plot), ch], alpha=0.7, linewidth=0.5,
                 label=f'DOF {ch + 1}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Input: Multi-channel Acceleration Data')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Natural frequencies
    ax2 = axes[0, 1]
    x_pos = np.arange(1, n_dof + 1)
    width = 0.35
    ax2.bar(x_pos - width / 2, freq_true, width, label='True', color='#2196F3', alpha=0.8)
    ax2.bar(x_pos + width / 2, freq_id, width, label='Identified (SSI)', color='#FF5722', alpha=0.8)
    ax2.set_xlabel('Mode Number')
    ax2.set_ylabel('Natural Frequency (Hz)')
    ax2.set_title('Natural Frequencies: True vs SSI-Identified')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for i in range(n_dof):
        re_val = metrics['freq_relative_errors'][i] * 100
        ax2.annotate(f'RE={re_val:.1f}%', xy=(x_pos[i], max(freq_true[i], freq_id[i])),
                     fontsize=7, ha='center', va='bottom')

    # Panel 3: Damping ratios
    ax3 = axes[1, 0]
    ax3.bar(x_pos - width / 2, zeta_true * 100, width, label='True', color='#4CAF50', alpha=0.8)
    ax3.bar(x_pos + width / 2, zeta_id * 100, width, label='Identified (SSI)', color='#FFC107', alpha=0.8)
    ax3.set_xlabel('Mode Number')
    ax3.set_ylabel('Damping Ratio (%)')
    ax3.set_title('Damping Ratios: True vs SSI-Identified')
    ax3.set_xticks(x_pos)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for i in range(n_dof):
        re_val = metrics['damping_relative_errors'][i] * 100
        ax3.annotate(f'RE={re_val:.1f}%', xy=(x_pos[i], max(zeta_true[i], zeta_id[i]) * 100),
                     fontsize=7, ha='center', va='bottom')

    # Panel 4: MAC matrix
    ax4 = axes[1, 1]
    im = ax4.imshow(mac_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    ax4.set_xlabel('Identified Mode')
    ax4.set_ylabel('True Mode')
    ax4.set_title('Modal Assurance Criterion (MAC) Matrix')
    ax4.set_xticks(range(n_dof))
    ax4.set_yticks(range(n_dof))
    ax4.set_xticklabels([f'M{i + 1}' for i in range(n_dof)])
    ax4.set_yticklabels([f'M{i + 1}' for i in range(n_dof)])
    for i in range(n_dof):
        for j in range(n_dof):
            color = 'white' if mac_matrix[i, j] > 0.5 else 'black'
            ax4.text(j, i, f'{mac_matrix[i, j]:.2f}', ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Stochastic Subspace Identification (Cov-SSI) — 5-DOF System\n'
        f'Mean Freq RE = {metrics["mean_freq_re"] * 100:.2f}%  |  '
        f'Mean Damping RE = {metrics["mean_damping_re"] * 100:.2f}%  |  '
        f'Mean MAC = {metrics["mean_mac"]:.4f}',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {vis_path}")

    return metrics
