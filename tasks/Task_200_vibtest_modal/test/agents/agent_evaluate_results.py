import os

import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(data, modal_params, H_recon):
    """
    Evaluate reconstruction quality and save results.
    
    Computes:
        - Frequency relative errors
        - Damping relative errors
        - MAC (Modal Assurance Criterion) matrix
        - FRF PSNR and correlation coefficient
    
    Args:
        data: dict from load_and_preprocess_data
        modal_params: dict from run_inversion
        H_recon: reconstructed FRF from forward_operator
    
    Returns:
        metrics: dict of evaluation metrics
    """
    print("\n[6] Reconstructing FRF...")
    print("\n[7] Evaluating...")
    
    omega = data['omega']
    H_true = data['H_true']
    H_noisy = data['H_noisy']
    omega_n = data['omega_n']
    zeta = data['zeta']
    Psi = data['Psi']
    
    freq_est = modal_params['freq_est']
    zeta_est = modal_params['zeta_est']
    phi_est = modal_params['phi_est']
    
    n_modes = len(omega_n)
    
    # Sort by frequency
    idx_true = np.argsort(omega_n)
    idx_est = np.argsort(freq_est)
    
    freq_true_s = omega_n[idx_true]
    freq_est_s = freq_est[idx_est]
    zeta_true_s = zeta[idx_true]
    zeta_est_s = zeta_est[idx_est]
    phi_true_s = Psi[:, idx_true]
    phi_est_s = phi_est[:, idx_est]
    
    # Frequency relative errors
    freq_re = np.abs(freq_est_s - freq_true_s) / freq_true_s
    
    # Damping relative errors
    damping_re = np.abs(zeta_est_s - zeta_true_s) / (zeta_true_s + 1e-10)
    
    # MAC matrix
    mac_matrix = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            num = np.abs(phi_true_s[:, i] @ phi_est_s[:, j])**2
            den = (phi_true_s[:, i] @ phi_true_s[:, i]) * (phi_est_s[:, j] @ phi_est_s[:, j])
            mac_matrix[i, j] = num / (den + 1e-30)
    mac_diag = np.diag(mac_matrix)
    
    # FRF PSNR
    mag_true = np.abs(H_true[:, 0])
    mag_recon = np.abs(H_recon[:, 0])
    mse = np.mean((mag_true - mag_recon)**2)
    max_val = np.max(mag_true)
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-30))
    
    # Correlation coefficient
    cc = float(np.corrcoef(mag_true, mag_recon)[0, 1])
    
    metrics = {
        "freq_relative_errors": freq_re.tolist(),
        "damping_relative_errors": damping_re.tolist(),
        "mac_values": mac_diag.tolist(),
        "mean_freq_re": float(np.mean(freq_re)),
        "mean_damping_re": float(np.mean(damping_re)),
        "mean_mac": float(np.mean(mac_diag)),
        "frf_psnr": float(psnr),
        "frf_cc": cc,
    }
    
    print("\n=== Evaluation Metrics ===")
    for i in range(n_modes):
        print(f"  Mode {i+1}: freq RE = {freq_re[i]:.4f}, "
              f"damping RE = {damping_re[i]:.4f}, MAC = {mac_diag[i]:.4f}")
    print(f"  Mean freq RE:    {metrics['mean_freq_re']:.4f}")
    print(f"  Mean damping RE: {metrics['mean_damping_re']:.4f}")
    print(f"  Mean MAC:        {metrics['mean_mac']:.4f}")
    print(f"  FRF PSNR:        {metrics['frf_psnr']:.2f} dB")
    print(f"  FRF CC:          {metrics['frf_cc']:.4f}")
    
    # Save outputs
    print("\n[8] Saving outputs...")
    gt_data = {
        "omega_n": omega_n, 
        "zeta": zeta, 
        "Psi": Psi,
        "H_true_real": np.real(H_true), 
        "H_true_imag": np.imag(H_true), 
        "omega": omega
    }
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data, allow_pickle=True)
    
    recon_data = {
        "freq_est": freq_est, 
        "zeta_est": zeta_est, 
        "phi_est": phi_est,
        "H_recon_real": np.real(H_recon), 
        "H_recon_imag": np.imag(H_recon)
    }
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_data, allow_pickle=True)
    
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot results
    print("\n[9] Plotting...")
    f_hz = omega / (2 * np.pi)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Task 200: Experimental Modal Analysis from FRF (3-DOF)\n"
        f"Mean Freq RE={metrics['mean_freq_re']:.4f} | "
        f"Mean Damping RE={metrics['mean_damping_re']:.4f} | "
        f"Mean MAC={metrics['mean_mac']:.4f} | "
        f"FRF PSNR={metrics['frf_psnr']:.1f} dB",
        fontsize=12, fontweight="bold"
    )
    
    # Panel 1: FRF magnitude
    ax = axes[0, 0]
    ax.semilogy(f_hz, np.abs(H_true[:, 0]), 'b-', lw=1.5, label='True FRF')
    ax.semilogy(f_hz, np.abs(H_noisy[:, 0]), 'gray', alpha=0.4, lw=0.5, label='Noisy FRF')
    ax.semilogy(f_hz, np.abs(H_recon[:, 0]), 'r--', lw=1.5, label='Reconstructed FRF')
    for fn in omega_n:
        ax.axvline(fn/(2*np.pi), color='blue', ls=':', alpha=0.3)
    for fn in freq_est:
        ax.axvline(fn/(2*np.pi), color='red', ls=':', alpha=0.3)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|H(ω)| (m/N)')
    ax.set_title('FRF Magnitude — DOF 1→1')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Natural frequencies comparison
    ax = axes[0, 1]
    x = np.arange(n_modes)
    width = 0.35
    ax.bar(x - width/2, freq_true_s/(2*np.pi), width,
           label='True', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, freq_est_s/(2*np.pi), width,
           label='Identified', color='coral', edgecolor='black')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Natural Frequency (Hz)')
    ax.set_title('Natural Frequencies')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Mode {i+1}' for i in range(n_modes)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Damping ratios comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, zeta_true_s*100, width,
           label='True', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, zeta_est_s*100, width,
           label='Identified', color='coral', edgecolor='black')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Damping Ratio (%)')
    ax.set_title('Damping Ratios')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Mode {i+1}' for i in range(n_modes)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: MAC matrix
    ax = axes[1, 1]
    im = ax.imshow(mac_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    for i in range(mac_matrix.shape[0]):
        for j in range(mac_matrix.shape[1]):
            ax.text(j, i, f'{mac_matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=10, color='white' if mac_matrix[i,j] > 0.5 else 'black')
    ax.set_xlabel('Identified Mode')
    ax.set_ylabel('True Mode')
    ax.set_title('MAC Matrix')
    ax.set_xticks(range(n_modes))
    ax.set_yticks(range(n_modes))
    ax.set_xticklabels([f'M{i+1}' for i in range(n_modes)])
    ax.set_yticklabels([f'M{i+1}' for i in range(n_modes)])
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {save_path}")
    
    return metrics
