"""
Task 200: vibtest_modal — Experimental Modal Analysis from FRF
==============================================================
Inverse Problem: Extract modal parameters (natural frequencies, damping ratios,
mode shapes) from Frequency Response Functions (FRF).

Pipeline:
    1. Define a known 3-DOF system (M, C, K matrices).
    2. Compute ground-truth modal parameters via eigenanalysis.
    3. Generate theoretical FRF.
    4. Add measurement noise to FRF.
    5. Extract modal parameters from noisy FRF (peak picking + half-power + shape extraction).
    6. Reconstruct FRF from identified parameters.
    7. Evaluate: Frequency RE, Damping RE, MAC, FRF PSNR/CC.

Repo: https://github.com/Vibration-Testing/vibrationtesting
Usage: /data/yjh/vibtest_modal_env/bin/python vibtest_modal_code.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# 1. Define 3-DOF system
# ======================================================================
def build_3dof_system():
    """3-DOF spring-mass-damper chain with well-separated modes."""
    m = np.array([2.0, 1.5, 1.0])
    M = np.diag(m)

    # Stiffness matrix (chain topology with ground springs)
    K = np.array([
        [2500.0, -2000.0,     0.0],
        [-2000.0, 3500.0, -1500.0],
        [    0.0, -1500.0, 2500.0],
    ])

    # Rayleigh damping: C = alpha*M + beta*K
    alpha = 0.5
    beta = 1e-4
    C = alpha * M + beta * K

    return M, C, K


def ground_truth_modal(M, C, K):
    """Compute ground truth modal parameters via eigenanalysis."""
    n = M.shape[0]
    # Solve generalized eigenvalue problem: K*phi = lambda*M*phi
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M, K))

    # Sort by frequency
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])

    omega_n = np.sqrt(eigenvalues)  # natural frequencies (rad/s)

    # Damping ratios from modal damping matrix
    # For Rayleigh damping: zeta_r = (alpha/(2*omega_r) + beta*omega_r/2)
    # where C = alpha*M + beta*K
    alpha = 0.5
    beta = 1e-4
    zeta = alpha / (2 * omega_n) + beta * omega_n / 2

    # Normalize mode shapes (mass-normalize)
    Psi = eigenvectors.copy()
    for i in range(n):
        modal_mass = Psi[:, i] @ M @ Psi[:, i]
        Psi[:, i] /= np.sqrt(abs(modal_mass))

    return omega_n, zeta, Psi


# ======================================================================
# 2. Generate FRF (Forward Operator)
# ======================================================================
def generate_frf(M, C, K, omega_low=0.5, omega_high=80.0, num_freqs=8000):
    """
    Generate FRF H(omega) for force at DOF 0, response at all DOFs.
    H_ij(omega) = [K - omega^2 * M + j*omega*C]^{-1}_{i,0}
    """
    omega = np.linspace(omega_low, omega_high, num_freqs)
    n = M.shape[0]
    H = np.zeros((num_freqs, n), dtype=complex)

    for k, w in enumerate(omega):
        Z = K - w**2 * M + 1j * w * C  # dynamic stiffness matrix
        Z_inv = np.linalg.inv(Z)
        H[k, :] = Z_inv[:, 0]  # force at DOF 0

    return omega, H


def add_noise_to_frf(H, snr_db=40):
    """Add complex Gaussian noise at specified SNR."""
    signal_power = np.mean(np.abs(H)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)
    )
    return H + noise


# ======================================================================
# 3. Modal Parameter Extraction (Inverse Solver)
# ======================================================================
def extract_modal_params(omega, H_noisy, n_modes=3):
    """
    Extract modal parameters using peak-picking + improved half-power bandwidth.
    Uses all FRF columns for mode shape extraction.
    """
    from scipy.signal import find_peaks as sp_find_peaks
    n_dof = H_noisy.shape[1]

    # --- Peak picking: use SUM of all FRF magnitudes (Mode Indicator Function) ---
    # This avoids anti-resonances that appear in individual FRFs
    mif = np.zeros(len(omega))
    for j in range(n_dof):
        mif += np.abs(H_noisy[:, j])**2
    mif_db = 10 * np.log10(mif + 1e-30)

    # Find peaks with adaptive prominence
    dyn_range = np.max(mif_db) - np.min(mif_db)
    prominence = 0.15 * dyn_range
    min_dist = max(int(0.03 * len(omega)), 10)

    peak_indices, props = sp_find_peaks(mif_db, prominence=prominence, distance=min_dist)

    # Sort by frequency and take first n_modes
    peak_indices = np.sort(peak_indices)[:n_modes]

    if len(peak_indices) < n_modes:
        # Lower threshold
        prominence = 0.05 * dyn_range
        peak_indices, _ = sp_find_peaks(mif_db, prominence=prominence, distance=min_dist)
        peak_indices = np.sort(peak_indices)[:n_modes]

    print(f"  Peak frequencies (Hz): {omega[peak_indices] / (2*np.pi)}")

    freq_est = np.zeros(n_modes)
    zeta_est = np.zeros(n_modes)
    phi_est = np.zeros((n_dof, n_modes))

    # Use drive-point FRF magnitude for half-power method
    mag = np.abs(H_noisy[:, 0])

    for r in range(n_modes):
        pk = peak_indices[r]
        omega_pk = omega[pk]

        # --- Refine peak location using parabolic interpolation ---
        if 1 <= pk < len(omega) - 1:
            alpha_val = mif_db[pk - 1]
            beta_val = mif_db[pk]
            gamma_val = mif_db[pk + 1]
            p = 0.5 * (alpha_val - gamma_val) / (alpha_val - 2*beta_val + gamma_val)
            domega = omega[1] - omega[0]
            omega_n_est = omega_pk + p * domega
        else:
            omega_n_est = omega_pk

        freq_est[r] = omega_n_est

        # --- Damping: half-power bandwidth method ---
        # Define search band
        bw_factor = 0.15
        if r > 0:
            f_lower = max(omega_n_est * (1 - bw_factor),
                         (omega[peak_indices[r-1]] + omega_n_est) / 2)
        else:
            f_lower = max(omega_n_est * (1 - bw_factor), omega[0])

        if r < n_modes - 1:
            f_upper = min(omega_n_est * (1 + bw_factor),
                         (omega_n_est + omega[peak_indices[r+1]]) / 2)
        else:
            f_upper = min(omega_n_est * (1 + bw_factor), omega[-1])

        mask = (omega >= f_lower) & (omega <= f_upper)
        omega_band = omega[mask]
        # Use the MIF (sum of squares) for half-power, as peaks were found there
        mif_band = np.sqrt(mif[mask])  # RMS of all FRFs

        peak_val_mif = np.sqrt(mif[pk])
        hp_level = peak_val_mif / np.sqrt(2)

        # Find crossing points
        above = mif_band > hp_level
        transitions = np.where(np.diff(above.astype(int)))[0]

        if len(transitions) >= 2:
            # Left crossing (first transition from below to above)
            idx_lo = transitions[0]
            # Interpolate
            if idx_lo + 1 < len(omega_band):
                frac = (hp_level - mif_band[idx_lo]) / (mif_band[idx_lo+1] - mif_band[idx_lo] + 1e-30)
                omega_1 = omega_band[idx_lo] + frac * (omega_band[idx_lo+1] - omega_band[idx_lo])
            else:
                omega_1 = omega_band[idx_lo]

            # Right crossing (last transition from above to below)
            idx_hi = transitions[-1]
            if idx_hi + 1 < len(omega_band):
                frac = (hp_level - mif_band[idx_hi]) / (mif_band[idx_hi+1] - mif_band[idx_hi] + 1e-30)
                omega_2 = omega_band[idx_hi] + frac * (omega_band[idx_hi+1] - omega_band[idx_hi])
            else:
                omega_2 = omega_band[idx_hi]

            zeta_est[r] = abs(omega_2 - omega_1) / (2 * omega_n_est)
        else:
            # Fallback: use Rayleigh formula estimate
            zeta_est[r] = 0.01

        zeta_est[r] = max(zeta_est[r], 1e-4)

        # --- Mode shape: use imaginary part of FRF at resonance ---
        # For lightly damped structures, the imaginary part of the FRF
        # at resonance is proportional to the mode shape (Ewins, Modal Testing)
        idx_res = np.argmin(np.abs(omega - freq_est[r]))
        
        # Use a small band around resonance for robustness
        bw_pts = max(int(0.005 * len(omega)), 3)
        lo = max(0, idx_res - bw_pts)
        hi = min(len(omega), idx_res + bw_pts + 1)
        
        for j in range(n_dof):
            # Average imaginary part around resonance
            imag_vals = np.imag(H_noisy[lo:hi, j])
            phi_est[j, r] = np.mean(imag_vals)
        
        # Sign convention: the sign of the imaginary part matters for mode shapes
        # Keep the sign (don't take abs) for proper MAC calculation

    # Normalize mode shapes
    for r in range(n_modes):
        norm = np.linalg.norm(phi_est[:, r])
        if norm > 1e-10:
            phi_est[:, r] /= norm

    return freq_est, zeta_est, phi_est


# ======================================================================
# 4. FRF Reconstruction
# ======================================================================
def reconstruct_frf(omega, freq_est, zeta_est, phi_est):
    """Reconstruct FRF from identified modal parameters."""
    n_dof = phi_est.shape[0]
    n_modes = phi_est.shape[1]
    N = len(omega)

    H_recon = np.zeros((N, n_dof), dtype=complex)
    for r in range(n_modes):
        wr = freq_est[r]
        zr = zeta_est[r]
        phi_r = phi_est[:, r]
        for i in range(n_dof):
            num = phi_r[i] * phi_r[0]
            denom = wr**2 - omega**2 + 2j * zr * wr * omega
            H_recon[:, i] += num / denom

    return H_recon


# ======================================================================
# 5. Evaluation
# ======================================================================
def compute_mac_matrix(phi_true, phi_est):
    """Compute MAC matrix between true and estimated mode shapes."""
    n_modes = phi_true.shape[1]
    mac = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            num = np.abs(phi_true[:, i] @ phi_est[:, j])**2
            den = (phi_true[:, i] @ phi_true[:, i]) * (phi_est[:, j] @ phi_est[:, j])
            mac[i, j] = num / (den + 1e-30)
    return mac


def compute_metrics(omega, H_true, H_recon, freq_true, freq_est,
                    zeta_true, zeta_est, phi_true, phi_est):
    """Compute all evaluation metrics."""
    n_modes = len(freq_true)

    # Sort by frequency
    idx_true = np.argsort(freq_true)
    idx_est = np.argsort(freq_est)

    freq_true_s = freq_true[idx_true]
    freq_est_s = freq_est[idx_est]
    zeta_true_s = zeta_true[idx_true]
    zeta_est_s = zeta_est[idx_est]
    phi_true_s = phi_true[:, idx_true]
    phi_est_s = phi_est[:, idx_est]

    # Frequency relative errors
    freq_re = np.abs(freq_est_s - freq_true_s) / freq_true_s

    # Damping relative errors
    damping_re = np.abs(zeta_est_s - zeta_true_s) / (zeta_true_s + 1e-10)

    # MAC matrix
    mac_matrix = compute_mac_matrix(phi_true_s, phi_est_s)
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

    return metrics, mac_matrix


# ======================================================================
# 6. Visualization
# ======================================================================
def plot_results(omega, H_true, H_noisy, H_recon,
                 freq_true, freq_est, zeta_true, zeta_est,
                 mac_matrix, metrics):
    f_hz = omega / (2 * np.pi)
    n_modes = len(freq_true)

    idx_true = np.argsort(freq_true)
    idx_est = np.argsort(freq_est)

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
    for fn in freq_true:
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
    ax.bar(x - width/2, freq_true[idx_true]/(2*np.pi), width,
           label='True', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, freq_est[idx_est]/(2*np.pi), width,
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
    ax.bar(x - width/2, zeta_true[idx_true]*100, width,
           label='True', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, zeta_est[idx_est]*100, width,
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


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 60)
    print("Task 200: vibtest_modal — Modal Analysis from FRF")
    print("=" * 60)

    np.random.seed(42)

    # Build system
    print("\n[1] Building 3-DOF system...")
    M, C, K = build_3dof_system()

    # Ground truth
    print("\n[2] Computing ground-truth modal parameters...")
    omega_n, zeta, Psi = ground_truth_modal(M, C, K)
    print(f"  Natural frequencies (Hz): {omega_n / (2*np.pi)}")
    print(f"  Damping ratios:           {zeta}")
    print(f"  Mode shapes:\n{Psi}")

    # Generate FRF
    print("\n[3] Generating theoretical FRF...")
    omega_high = np.max(omega_n) * 1.5
    omega, H_true = generate_frf(M, C, K, omega_low=0.5,
                                  omega_high=omega_high, num_freqs=8000)
    print(f"  Freq range: {omega[0]/(2*np.pi):.2f} – {omega[-1]/(2*np.pi):.2f} Hz")

    # Add noise (high SNR for better extraction)
    print("\n[4] Adding noise (SNR = 40 dB)...")
    H_noisy = add_noise_to_frf(H_true, snr_db=40)

    # Extract modal parameters
    print("\n[5] Extracting modal parameters...")
    freq_est, zeta_est, phi_est = extract_modal_params(omega, H_noisy, n_modes=3)
    print(f"  Identified frequencies (Hz): {freq_est / (2*np.pi)}")
    print(f"  Identified damping ratios:   {zeta_est}")

    # Reconstruct FRF
    print("\n[6] Reconstructing FRF...")
    H_recon = reconstruct_frf(omega, freq_est, zeta_est, phi_est)

    # Evaluate
    print("\n[7] Evaluating...")
    metrics, mac_matrix = compute_metrics(
        omega, H_true, H_recon, omega_n, freq_est, zeta, zeta_est, Psi, phi_est
    )

    # Save
    print("\n[8] Saving outputs...")
    gt_data = {"omega_n": omega_n, "zeta": zeta, "Psi": Psi,
               "H_true_real": np.real(H_true), "H_true_imag": np.imag(H_true), "omega": omega}
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data, allow_pickle=True)

    recon_data = {"freq_est": freq_est, "zeta_est": zeta_est, "phi_est": phi_est,
                  "H_recon_real": np.real(H_recon), "H_recon_imag": np.imag(H_recon)}
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_data, allow_pickle=True)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    print("\n[9] Plotting...")
    plot_results(omega, H_true, H_noisy, H_recon,
                 omega_n, freq_est, zeta, zeta_est, mac_matrix, metrics)

    print("\n✓ Task 200 complete.")


if __name__ == "__main__":
    main()
