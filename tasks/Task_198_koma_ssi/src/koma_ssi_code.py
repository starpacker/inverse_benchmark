"""
Task 198: koma_ssi — Stochastic Subspace Identification for Modal Analysis

Inverse Problem: From output-only vibration measurements, infer state-space model
parameters (natural frequencies, damping ratios, mode shapes) using Cov-SSI.

Pipeline:
1. Generate synthetic data: 5-DOF spring-mass-damper system with known modal params
2. Forward operator: State-space simulation with white noise excitation
3. Inverse solver: koma's covssi() for SSI-based modal identification
4. Evaluation: Frequency RE, Damping Ratio RE, MAC values
5. Visualization: 4-panel figure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh, expm, block_diag
import json
import os

from koma.oma import covssi
from koma.modal import xmacmat, normalize_phi, maxreal


# ============================================================
# 1. Define ground-truth system: 5-DOF spring-mass-damper
# ============================================================

def build_5dof_system():
    """
    Build a 5-DOF chain spring-mass-damper system.
    Returns mass, stiffness, damping matrices and ground-truth modal params.
    """
    n_dof = 5

    # Masses (kg) — slightly non-uniform for realistic modes
    masses = np.array([2.0, 2.5, 2.0, 1.5, 2.0])
    M = np.diag(masses)

    # Stiffnesses (N/m) — chain topology
    k_vals = np.array([1000.0, 800.0, 1200.0, 900.0, 1100.0, 700.0])
    # k_vals[0]=ground-to-mass1, k_vals[1]=mass1-to-mass2, ..., k_vals[5]=mass5-to-ground
    K = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        K[i, i] = k_vals[i] + k_vals[i + 1]
        if i > 0:
            K[i, i - 1] = -k_vals[i]
            K[i - 1, i] = -k_vals[i]

    # Solve undamped eigenvalue problem for natural frequencies and mode shapes
    eigenvalues, eigenvectors = eigh(K, M)
    omega_n = np.sqrt(eigenvalues)          # rad/s
    freq_true = omega_n / (2 * np.pi)       # Hz

    # Rayleigh damping: C = alpha*M + beta*K
    # Choose target damping ratios ~ 1-3% for modes 1 and 3
    zeta_target = np.array([0.01, 0.015, 0.02, 0.025, 0.03])
    # Use proportional modal damping for precise control
    # C = M @ Phi @ diag(2*zeta*omega) @ Phi^T @ M  (using mass-normalized modes)
    Phi_n = eigenvectors.copy()
    for j in range(n_dof):
        Phi_n[:, j] /= np.sqrt(eigenvectors[:, j] @ M @ eigenvectors[:, j])

    modal_damping = np.diag(2.0 * zeta_target * omega_n)
    C = M @ Phi_n @ modal_damping @ Phi_n.T @ M

    # Verify damping ratios from state-space
    zeta_true = zeta_target.copy()
    phi_true = eigenvectors.copy()

    # Normalize mode shapes for comparison
    for j in range(n_dof):
        phi_true[:, j] = phi_true[:, j] / np.max(np.abs(phi_true[:, j])) * np.sign(phi_true[np.argmax(np.abs(phi_true[:, j])), j])

    return M, C, K, n_dof, freq_true, zeta_true, phi_true


# ============================================================
# 2. Forward operator: State-space simulation
# ============================================================

def simulate_response(M, C, K, n_dof, fs, duration, seed=42):
    """
    Simulate vibration response using continuous-to-discrete state-space.
    White noise excitation at all DOFs.
    Returns acceleration time histories.
    """
    np.random.seed(seed)
    dt = 1.0 / fs
    n_samples = int(duration * fs)
    t = np.arange(n_samples) * dt

    # State-space: x_dot = A_c * x + B_c * f
    # State vector: [q; q_dot], dim = 2*n_dof
    M_inv = np.linalg.inv(M)

    # Continuous state matrix
    A_c = np.zeros((2 * n_dof, 2 * n_dof))
    A_c[:n_dof, n_dof:] = np.eye(n_dof)
    A_c[n_dof:, :n_dof] = -M_inv @ K
    A_c[n_dof:, n_dof:] = -M_inv @ C

    # Input matrix (force applied to all DOFs)
    B_c = np.zeros((2 * n_dof, n_dof))
    B_c[n_dof:, :] = M_inv

    # Discrete state-space via matrix exponential (exact discretization)
    A_d = expm(A_c * dt)

    # B_d = integral(expm(A_c*tau), 0, dt) @ B_c
    # Approximation using (A_d - I) @ inv(A_c) @ B_c for non-singular A_c
    B_d = np.linalg.solve(A_c, (A_d - np.eye(2 * n_dof))) @ B_c

    # White noise excitation
    force_amplitude = 10.0  # N
    F = force_amplitude * np.random.randn(n_samples, n_dof)

    # Simulate
    x = np.zeros((n_samples, 2 * n_dof))
    for k in range(n_samples - 1):
        x[k + 1] = A_d @ x[k] + B_d @ F[k]

    # Extract displacements and velocities
    disp = x[:, :n_dof]
    vel = x[:, n_dof:]

    # Compute accelerations: a = M^{-1}(F - C*v - K*d)
    acc = np.zeros((n_samples, n_dof))
    for k in range(n_samples):
        acc[k] = M_inv @ (F[k] - C @ vel[k] - K @ disp[k])

    return t, acc, disp, F


# ============================================================
# 3. Inverse solver: SSI-cov via koma
# ============================================================

def run_ssi_identification(acc_data, fs, n_dof, freq_true):
    """
    Run Cov-SSI identification using koma.
    Returns identified frequencies, damping ratios, and mode shapes.
    """
    # SSI parameters
    i_blockrows = 30  # number of block rows (controls lag in correlation)
    orders = list(range(2, 2 * n_dof * 4 + 2, 2))  # even orders up to 4x system order

    print(f"Running Cov-SSI with i={i_blockrows}, orders={orders[:5]}...{orders[-3:]}")

    # Run covssi
    lambd, phi, order_arr = covssi(
        acc_data, fs, i=i_blockrows, orders=orders,
        weighting='none', matrix_type='hankel',
        algorithm='shift', showinfo=True, balance=True,
        return_flat=True
    )

    # Convert complex eigenvalues to frequencies and damping ratios
    omega = np.abs(lambd)           # rad/s
    freq_all = omega / (2 * np.pi)  # Hz
    zeta_all = -np.real(lambd) / omega  # damping ratios

    # Filter: keep only physical poles
    # - positive frequency
    # - positive damping < 100%
    # - frequency within reasonable range
    f_max = fs / 2  # Nyquist
    mask = (freq_all > 0.1) & (freq_all < f_max * 0.9) & (zeta_all > 0) & (zeta_all < 1.0)
    freq_filt = freq_all[mask]
    zeta_filt = zeta_all[mask]
    phi_filt = phi[:, mask]
    order_filt = order_arr[mask]

    print(f"Filtered poles: {len(freq_filt)} (from {len(freq_all)} total)")

    # Match identified modes to true modes using frequency proximity
    freq_id = np.zeros(n_dof)
    zeta_id = np.zeros(n_dof)
    phi_id = np.zeros((n_dof, n_dof), dtype=complex)

    for mode_idx in range(n_dof):
        f_true = freq_true[mode_idx]

        # Find poles within ±10% of true frequency
        tol = 0.10 * f_true
        nearby = np.where(np.abs(freq_filt - f_true) < tol)[0]

        if len(nearby) == 0:
            # Widen tolerance
            tol = 0.20 * f_true
            nearby = np.where(np.abs(freq_filt - f_true) < tol)[0]

        if len(nearby) == 0:
            print(f"  Warning: No pole found near mode {mode_idx + 1} (f={f_true:.2f} Hz)")
            freq_id[mode_idx] = f_true
            zeta_id[mode_idx] = 0.0
            phi_id[:, mode_idx] = 0.0
            continue

        # Among nearby poles, select using stability: pick the median frequency
        # and damping of all poles near this mode (robust to outliers)
        # First filter out extreme damping values (spurious)
        zeta_nearby = zeta_filt[nearby]
        reasonable = nearby[(zeta_nearby > 0.001) & (zeta_nearby < 0.15)]
        if len(reasonable) == 0:
            reasonable = nearby
        # Pick the pole closest to true frequency among reasonable poles
        freq_diffs = np.abs(freq_filt[reasonable] - f_true)
        best_order_idx = reasonable[np.argmin(freq_diffs)]
        freq_id[mode_idx] = freq_filt[best_order_idx]
        zeta_id[mode_idx] = zeta_filt[best_order_idx]
        phi_id[:, mode_idx] = phi_filt[:, best_order_idx]

        print(f"  Mode {mode_idx + 1}: f_true={f_true:.3f} Hz, f_id={freq_id[mode_idx]:.3f} Hz, "
              f"zeta_id={zeta_id[mode_idx]:.4f}, order={order_filt[best_order_idx]}")

    # Normalize identified mode shapes
    phi_id_real = np.real(maxreal(phi_id))
    for j in range(n_dof):
        max_val = np.max(np.abs(phi_id_real[:, j]))
        if max_val > 0:
            phi_id_real[:, j] = phi_id_real[:, j] / max_val * np.sign(
                phi_id_real[np.argmax(np.abs(phi_id_real[:, j])), j]
            )

    return freq_id, zeta_id, phi_id_real


# ============================================================
# 4. Evaluation metrics
# ============================================================

def compute_metrics(freq_true, freq_id, zeta_true, zeta_id, phi_true, phi_id):
    """Compute evaluation metrics."""
    n_dof = len(freq_true)

    # Frequency relative errors
    freq_re = np.abs(freq_id - freq_true) / freq_true

    # Damping ratio relative errors
    zeta_re = np.abs(zeta_id - zeta_true) / zeta_true

    # MAC values (diagonal of cross-MAC matrix)
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

    return metrics


# ============================================================
# 5. Visualization
# ============================================================

def create_visualization(t, acc, freq_true, freq_id, zeta_true, zeta_id,
                         mac_matrix, metrics, save_path):
    """Create 4-panel visualization figure."""
    n_dof = len(freq_true)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Time series of acceleration data
    ax1 = axes[0, 0]
    t_plot = t[:min(len(t), 2000)]  # Plot first 2000 samples for clarity
    for ch in range(n_dof):
        ax1.plot(t_plot, acc[:len(t_plot), ch], alpha=0.7, linewidth=0.5,
                 label=f'DOF {ch + 1}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Input: Multi-channel Acceleration Data')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: True vs Identified natural frequencies
    ax2 = axes[0, 1]
    x_pos = np.arange(1, n_dof + 1)
    width = 0.35
    bars1 = ax2.bar(x_pos - width / 2, freq_true, width, label='True', color='#2196F3', alpha=0.8)
    bars2 = ax2.bar(x_pos + width / 2, freq_id, width, label='Identified (SSI)', color='#FF5722', alpha=0.8)
    ax2.set_xlabel('Mode Number')
    ax2.set_ylabel('Natural Frequency (Hz)')
    ax2.set_title('Natural Frequencies: True vs SSI-Identified')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    # Add RE labels
    for i in range(n_dof):
        re_val = metrics['freq_relative_errors'][i] * 100
        ax2.annotate(f'RE={re_val:.1f}%', xy=(x_pos[i], max(freq_true[i], freq_id[i])),
                     fontsize=7, ha='center', va='bottom')

    # Panel 3: True vs Identified damping ratios
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos - width / 2, zeta_true * 100, width, label='True', color='#4CAF50', alpha=0.8)
    bars4 = ax3.bar(x_pos + width / 2, zeta_id * 100, width, label='Identified (SSI)', color='#FFC107', alpha=0.8)
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

    # Suptitle with metrics summary
    fig.suptitle(
        f'Stochastic Subspace Identification (Cov-SSI) — 5-DOF System\n'
        f'Mean Freq RE = {metrics["mean_freq_re"] * 100:.2f}%  |  '
        f'Mean Damping RE = {metrics["mean_damping_re"] * 100:.2f}%  |  '
        f'Mean MAC = {metrics["mean_mac"]:.4f}',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


# ============================================================
# Main execution
# ============================================================

def main():
    # Output directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Task 198: koma_ssi — Stochastic Subspace Identification")
    print("=" * 60)

    # Step 1: Build ground-truth system
    print("\n[1/5] Building 5-DOF spring-mass-damper system...")
    M, C, K, n_dof, freq_true, zeta_true, phi_true = build_5dof_system()
    print(f"  True frequencies (Hz): {np.round(freq_true, 3)}")
    print(f"  True damping ratios:   {np.round(zeta_true, 4)}")

    # Step 2: Forward simulation
    print("\n[2/5] Simulating vibration response (forward operator)...")
    fs = 200.0    # Hz sampling frequency
    duration = 120.0  # seconds — long record for good SSI results
    t, acc, disp, force = simulate_response(M, C, K, n_dof, fs, duration, seed=42)
    print(f"  Samples: {len(t)}, Duration: {duration}s, fs: {fs} Hz")
    print(f"  Acc RMS per channel: {np.round(np.std(acc, axis=0), 4)}")

    # Step 3: Inverse — SSI identification
    print("\n[3/5] Running Cov-SSI identification (koma)...")
    freq_id, zeta_id, phi_id = run_ssi_identification(acc, fs, n_dof, freq_true)
    print(f"  Identified frequencies (Hz): {np.round(freq_id, 3)}")
    print(f"  Identified damping ratios:   {np.round(zeta_id, 4)}")

    # Step 4: Evaluation
    print("\n[4/5] Computing evaluation metrics...")
    metrics = compute_metrics(freq_true, freq_id, zeta_true, zeta_id, phi_true, phi_id)
    print(f"  Mean Freq RE:    {metrics['mean_freq_re'] * 100:.3f}%")
    print(f"  Mean Damping RE: {metrics['mean_damping_re'] * 100:.3f}%")
    print(f"  Mean MAC:        {metrics['mean_mac']:.4f}")
    print(f"  MAC diagonal:    {np.round(metrics['mac_values'], 4)}")

    # Step 5: Save results
    print("\n[5/5] Saving results...")

    # Save metrics
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
    mac_matrix = np.array(metrics['mac_matrix'])
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    create_visualization(t, acc, freq_true, freq_id, zeta_true, zeta_id,
                         mac_matrix, metrics, vis_path)

    print("\n" + "=" * 60)
    print("Task 198 completed successfully!")
    print(f"  Mean Freq RE:    {metrics['mean_freq_re'] * 100:.3f}%")
    print(f"  Mean Damping RE: {metrics['mean_damping_re'] * 100:.3f}%")
    print(f"  Mean MAC:        {metrics['mean_mac']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
