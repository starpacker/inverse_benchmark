import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as sp_find_peaks

# Import the target function
from agent_run_inversion import run_inversion

# Define RESULTS_DIR for evaluate_results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the evaluate_results function (The Referee)
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
    
    return metrics


def forward_operator(omega, freq_est, zeta_est, phi_est):
    """
    Reconstruct FRF from identified modal parameters.
    
    Args:
        omega: frequency array (rad/s)
        freq_est: estimated natural frequencies (rad/s)
        zeta_est: estimated damping ratios
        phi_est: estimated mode shapes (n_dof x n_modes)
    
    Returns:
        H_recon: reconstructed FRF (n_freq x n_dof)
    """
    n_freq = len(omega)
    n_dof = phi_est.shape[0]
    n_modes = len(freq_est)
    
    H_recon = np.zeros((n_freq, n_dof), dtype=complex)
    
    for r in range(n_modes):
        omega_n = freq_est[r]
        zeta_r = zeta_est[r]
        phi_r = phi_est[:, r]
        
        # Modal participation (assuming unit modal mass)
        for j in range(n_dof):
            # FRF contribution from mode r
            # H_jk = sum_r (phi_jr * phi_kr) / (omega_n^2 - omega^2 + 2j*zeta*omega_n*omega)
            for i_freq, w in enumerate(omega):
                denom = omega_n**2 - w**2 + 2j * zeta_r * omega_n * w
                # Assuming excitation at DOF 0
                H_recon[i_freq, j] += phi_r[j] * phi_r[0] / denom
    
    return H_recon


def main():
    # Data paths
    data_paths = ['/data/yjh/vibtest_modal_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer data
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Loaded outer data with keys: {outer_data.keys()}")
        
        args = outer_data['args']
        kwargs = outer_data['kwargs']
        std_output = outer_data['output']
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys() if kwargs else 'None'}")
        
        # Execute the agent's run_inversion
        print("\n=== Running Agent's run_inversion ===")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if inner_data_paths:
            # Chained execution pattern
            print("\n=== Detected Chained Execution Pattern ===")
            for inner_path in inner_data_paths:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data['args']
                inner_kwargs = inner_data['kwargs']
                std_inner_output = inner_data['output']
                
                # Execute the operator returned by run_inversion
                final_agent_result = agent_output(*inner_args, **inner_kwargs)
                final_std_result = std_inner_output
        else:
            # Direct execution pattern
            print("\n=== Direct Execution Pattern ===")
            final_agent_result = agent_output
            final_std_result = std_output
        
        # Now we need to evaluate. The evaluate_results function requires:
        # - data: dict with omega, H_true, H_noisy, omega_n, zeta, Psi
        # - modal_params: dict with freq_est, zeta_est, phi_est
        # - H_recon: reconstructed FRF
        
        # Extract omega and H_noisy from args
        omega = args[0]  # First argument is omega
        H_noisy = args[1]  # Second argument is H_noisy
        n_modes = kwargs.get('n_modes', 3) if kwargs else 3
        
        print(f"\nomega shape: {omega.shape}")
        print(f"H_noisy shape: {H_noisy.shape}")
        print(f"n_modes: {n_modes}")
        
        # For evaluation, we need the ground truth data
        # This should be available in the test environment or we reconstruct from available info
        # Since we don't have explicit ground truth, we'll need to create synthetic data
        # or load it from another source
        
        # Let's check if there's additional data we can use
        # Looking at the original code, the ground truth should be available
        
        # Create a data dict for evaluation
        # We'll need to extract or estimate the ground truth
        # For now, let's use the standard output to get ground truth estimates
        
        # The standard output contains: freq_est, zeta_est, phi_est
        freq_est_std = final_std_result['freq_est']
        zeta_est_std = final_std_result['zeta_est']
        phi_est_std = final_std_result['phi_est']
        
        # Agent output
        freq_est_agent = final_agent_result['freq_est']
        zeta_est_agent = final_agent_result['zeta_est']
        phi_est_agent = final_agent_result['phi_est']
        
        print("\n=== Standard Output ===")
        print(f"freq_est: {freq_est_std}")
        print(f"zeta_est: {zeta_est_std}")
        print(f"phi_est shape: {phi_est_std.shape}")
        
        print("\n=== Agent Output ===")
        print(f"freq_est: {freq_est_agent}")
        print(f"zeta_est: {zeta_est_agent}")
        print(f"phi_est shape: {phi_est_agent.shape}")
        
        # Compute comparison metrics directly
        # Frequency comparison
        freq_diff = np.abs(freq_est_agent - freq_est_std)
        freq_rel_diff = freq_diff / (np.abs(freq_est_std) + 1e-10)
        mean_freq_rel_diff = np.mean(freq_rel_diff)
        
        # Damping comparison
        zeta_diff = np.abs(zeta_est_agent - zeta_est_std)
        zeta_rel_diff = zeta_diff / (np.abs(zeta_est_std) + 1e-10)
        mean_zeta_rel_diff = np.mean(zeta_rel_diff)
        
        # Mode shape comparison using MAC
        n_modes_actual = len(freq_est_std)
        mac_diag = []
        for r in range(n_modes_actual):
            phi_std = phi_est_std[:, r]
            phi_agent = phi_est_agent[:, r]
            num = np.abs(np.dot(phi_std, phi_agent))**2
            den = np.dot(phi_std, phi_std) * np.dot(phi_agent, phi_agent)
            mac_val = num / (den + 1e-30)
            mac_diag.append(mac_val)
        mean_mac = np.mean(mac_diag)
        
        print("\n=== Comparison Metrics ===")
        print(f"Frequency relative differences: {freq_rel_diff}")
        print(f"Mean frequency relative difference: {mean_freq_rel_diff:.6f}")
        print(f"Damping relative differences: {zeta_rel_diff}")
        print(f"Mean damping relative difference: {mean_zeta_rel_diff:.6f}")
        print(f"MAC values (diagonal): {mac_diag}")
        print(f"Mean MAC: {mean_mac:.6f}")
        
        # Reconstruct FRF for both and compare
        H_recon_std = forward_operator(omega, freq_est_std, zeta_est_std, phi_est_std)
        H_recon_agent = forward_operator(omega, freq_est_agent, zeta_est_agent, phi_est_agent)
        
        # PSNR between reconstructed FRFs
        mag_std = np.abs(H_recon_std[:, 0])
        mag_agent = np.abs(H_recon_agent[:, 0])
        mse = np.mean((mag_std - mag_agent)**2)
        max_val = np.max(mag_std)
        if mse > 0:
            psnr = 10 * np.log10(max_val**2 / mse)
        else:
            psnr = float('inf')
        
        # Correlation coefficient
        cc = np.corrcoef(mag_std, mag_agent)[0, 1]
        
        print(f"\nFRF comparison (Agent vs Standard):")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Correlation: {cc:.6f}")
        
        # Define success criteria
        # For modal analysis, we want:
        # - Frequencies to be within 5% relative error
        # - Damping to be within 50% relative error (damping is harder to estimate)
        # - MAC to be above 0.9 (high similarity)
        
        success = True
        tolerance_freq = 0.05  # 5% tolerance for frequency
        tolerance_zeta = 0.5   # 50% tolerance for damping
        tolerance_mac = 0.9    # MAC should be above 0.9
        tolerance_cc = 0.95    # Correlation should be above 0.95
        
        print("\n=== Validation Results ===")
        
        if mean_freq_rel_diff > tolerance_freq:
            print(f"WARNING: Mean frequency relative difference ({mean_freq_rel_diff:.4f}) exceeds tolerance ({tolerance_freq})")
            # Only fail if significantly worse
            if mean_freq_rel_diff > tolerance_freq * 2:
                success = False
        else:
            print(f"PASS: Frequency estimation within tolerance")
        
        if mean_zeta_rel_diff > tolerance_zeta:
            print(f"WARNING: Mean damping relative difference ({mean_zeta_rel_diff:.4f}) exceeds tolerance ({tolerance_zeta})")
            # Only fail if significantly worse
            if mean_zeta_rel_diff > tolerance_zeta * 2:
                success = False
        else:
            print(f"PASS: Damping estimation within tolerance")
        
        if mean_mac < tolerance_mac:
            print(f"WARNING: Mean MAC ({mean_mac:.4f}) below tolerance ({tolerance_mac})")
            if mean_mac < tolerance_mac * 0.8:
                success = False
        else:
            print(f"PASS: Mode shape correlation (MAC) within tolerance")
        
        if cc < tolerance_cc:
            print(f"WARNING: FRF correlation ({cc:.4f}) below tolerance ({tolerance_cc})")
            if cc < tolerance_cc * 0.9:
                success = False
        else:
            print(f"PASS: FRF correlation within tolerance")
        
        print(f"\n=== Final Score ===")
        print(f"Scores -> Agent freq_error: {mean_freq_rel_diff:.6f}, Standard freq_error: 0.0")
        print(f"Scores -> Agent zeta_error: {mean_zeta_rel_diff:.6f}, Standard zeta_error: 0.0")
        print(f"Scores -> Agent MAC: {mean_mac:.6f}, Standard MAC: 1.0")
        print(f"Scores -> Agent CC: {cc:.6f}, Standard CC: 1.0")
        
        if success:
            print("\n=== TEST PASSED ===")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()