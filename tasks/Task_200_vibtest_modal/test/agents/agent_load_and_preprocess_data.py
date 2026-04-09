import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(seed=42, snr_db=40):
    """
    Load and preprocess data for modal analysis.
    
    1. Define a known 3-DOF system (M, C, K matrices).
    2. Compute ground-truth modal parameters via eigenanalysis.
    3. Generate theoretical FRF.
    4. Add measurement noise to FRF.
    
    Returns:
        dict containing:
            - M, C, K: system matrices
            - omega_n, zeta, Psi: ground truth modal parameters
            - omega: frequency array (rad/s)
            - H_true: true FRF (complex)
            - H_noisy: noisy FRF (complex)
    """
    np.random.seed(seed)
    
    # Build 3-DOF system
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
    
    # Compute ground truth modal parameters via eigenanalysis
    n = M.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M, K))
    
    # Sort by frequency
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])
    
    omega_n = np.sqrt(eigenvalues)  # natural frequencies (rad/s)
    
    # Damping ratios from modal damping matrix
    zeta = alpha / (2 * omega_n) + beta * omega_n / 2
    
    # Normalize mode shapes (mass-normalize)
    Psi = eigenvectors.copy()
    for i in range(n):
        modal_mass = Psi[:, i] @ M @ Psi[:, i]
        Psi[:, i] /= np.sqrt(abs(modal_mass))
    
    # Generate FRF
    omega_low = 0.5
    omega_high = np.max(omega_n) * 1.5
    num_freqs = 8000
    omega = np.linspace(omega_low, omega_high, num_freqs)
    
    H_true = np.zeros((num_freqs, n), dtype=complex)
    for k, w in enumerate(omega):
        Z = K - w**2 * M + 1j * w * C  # dynamic stiffness matrix
        Z_inv = np.linalg.inv(Z)
        H_true[k, :] = Z_inv[:, 0]  # force at DOF 0
    
    # Add noise
    signal_power = np.mean(np.abs(H_true)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*H_true.shape) + 1j * np.random.randn(*H_true.shape)
    )
    H_noisy = H_true + noise
    
    print("\n[1] Building 3-DOF system...")
    print("\n[2] Computing ground-truth modal parameters...")
    print(f"  Natural frequencies (Hz): {omega_n / (2*np.pi)}")
    print(f"  Damping ratios:           {zeta}")
    print(f"  Mode shapes:\n{Psi}")
    print("\n[3] Generating theoretical FRF...")
    print(f"  Freq range: {omega[0]/(2*np.pi):.2f} – {omega[-1]/(2*np.pi):.2f} Hz")
    print(f"\n[4] Adding noise (SNR = {snr_db} dB)...")
    
    return {
        'M': M,
        'C': C,
        'K': K,
        'omega_n': omega_n,
        'zeta': zeta,
        'Psi': Psi,
        'omega': omega,
        'H_true': H_true,
        'H_noisy': H_noisy,
    }
