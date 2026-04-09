import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os
import mne
from mne.datasets import sample

# -----------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# -----------------------------------------------------------------------------
def load_and_preprocess_data():
    """
    Loads sample data, computes noise covariance, and prepares matrices for inversion.
    
    Returns:
        tuple: (G, C, y, nave, P, info, evoked_ref)
            - G (np.ndarray): Gain matrix (n_channels, n_sources)
            - C (np.ndarray): Noise covariance (n_channels, n_channels)
            - y (np.ndarray): Sensor data (n_channels, n_times)
            - nave (int): Number of averages
            - P (np.ndarray): Projection matrix (n_channels, n_channels)
            - info (mne.Info): Measurement info
            - evoked_ref (mne.Evoked): Reference evoked object for MNE comparison
    """
    print("\n=== Phase 1: Data Preprocessing & Forward Model (MNE) ===")
    data_path = sample.data_path()
    raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    event_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw-eve.fif')
    fwd_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-eeg-oct-6-fwd.fif')
    
    # Read Raw
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    events = mne.read_events(event_fname)
    
    # Pick MEG channels
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, exclude='bads')
    
    # Create Epochs (Left Auditory)
    event_id, tmin, tmax = 1, -0.2, 0.5
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
    
    # Average epochs (removed unsupported 'verbose' kwarg)
    evoked = epochs.average()
    
    # Compute Noise Covariance
    noise_cov = mne.compute_covariance(epochs, tmax=0, method=['shrunk', 'empirical'])
    
    # Read Forward Solution
    forward = mne.read_forward_solution(fwd_fname)
    
    # Convert to FIXED orientation (simplifies the problem to 1 source per vertex)
    forward = mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True, use_cps=True)
    forward = mne.pick_types_forward(forward, meg=True, eeg=False)
    
    # Ensure evoked picks match forward
    evoked.pick_types(meg=True, eeg=False)
    
    # Intersect channels between Info, Forward, and Covariance
    info = evoked.info
    common_channels = [ch for ch in info['ch_names'] 
                       if ch in forward['info']['ch_names'] 
                       and ch in noise_cov['names']]
    print(f"Number of common channels: {len(common_channels)}")
    
    # Subset and reorder
    evoked.pick_channels(common_channels)
    forward = mne.pick_channels_forward(forward, common_channels)
    noise_cov = mne.pick_channels_cov(noise_cov, common_channels)
    
    # Extract numpy arrays
    y = evoked.data  # (n_channels, n_times)
    G = forward['sol']['data']  # (n_channels, n_sources)
    C = noise_cov['data']  # (n_channels, n_channels)
    nave = evoked.nave
    
    # --- Compute Projection Matrix P ---
    def compute_proj_matrix(projs, ch_names):
        n_ch = len(ch_names)
        P_out = np.eye(n_ch)
        all_vectors = []
        for p in projs:
            if p['active']:
                proj_ch_names = p['data']['col_names']
                vecs = p['data']['data']
                for v in vecs:
                    full_v = np.zeros(n_ch)
                    for i, ch in enumerate(proj_ch_names):
                        if ch in ch_names:
                            idx = ch_names.index(ch)
                            full_v[idx] = v[i]
                    all_vectors.append(full_v)
        
        if not all_vectors:
            return P_out
            
        U_mat = np.array(all_vectors).T
        Q, _ = scipy.linalg.qr(U_mat, mode='economic')
        P_out = np.eye(n_ch) - np.dot(Q, Q.T)
        return P_out

    P = compute_proj_matrix(info['projs'], common_channels)
    
    # Also return the Forward object and NoiseCov object for the reference MNE run
    # Pack these into a wrapper or simple struct if needed, but here we just attach 
    # them to the evoked object or return them separately.
    # We will return 'evoked' (which has info) and 'forward', 'noise_cov' separately 
    # for the evaluation phase.
    
    # To strictly follow the signature requested, we return the matrices for the standalone solver
    # and objects for the MNE reference solver.
    return G, C, y, nave, P, info, evoked, forward, noise_cov


# -----------------------------------------------------------------------------
# 2. Forward Operator
# -----------------------------------------------------------------------------
def forward_operator(x, G):
    """
    Computes the forward model y_pred = G * x.
    This simulates sensor data from source currents.
    
    Args:
        x (np.ndarray): Source currents (n_sources, n_times)
        G (np.ndarray): Gain matrix (n_channels, n_sources)
        
    Returns:
        y_pred (np.ndarray): Simulated sensor data (n_channels, n_times)
    """
    # Simple matrix multiplication representing the physical forward model
    y_pred = np.dot(G, x)
    return y_pred


# -----------------------------------------------------------------------------
# 3. Run Inversion
# -----------------------------------------------------------------------------
def run_inversion(G, C, y, nave, P, snr=3.0, method='dSPM'):
    """
    Computes the inverse operator K and applies it to sensor data y.
    
    Mathematical Steps:
    1. C_scaled = C / nave
    2. Project C and G using P
    3. Whiten G -> G_tilde
    4. SVD of G_tilde
    5. Compute K
    6. Apply K to y
    7. Apply noise normalization (if dSPM)
    
    Returns:
        source_estimate (np.ndarray): (n_sources, n_times)
    """
    print("\n--- [Standalone] Fitting Inverse Model ---")
    lambda2 = 1.0 / (snr ** 2)
    
    # 0. Adjust Noise Covariance for Averaging
    C_scaled = C / nave
    
    # 1. Apply Projections
    # C_proj = P * C_scaled * P^T
    C_proj = np.dot(P, np.dot(C_scaled, P.T))
    # G_proj = P * G
    G_proj = np.dot(P, G)
    
    # 2. Compute Whitener W = C^(-1/2) using eigh
    eig_vals, eig_vecs = scipy.linalg.eigh(C_proj)
    
    # Filter small eigenvalues
    max_eig = np.max(eig_vals)
    tol = 1e-6 * max_eig
    mask = eig_vals > tol
    
    print(f"Rank of Noise Covariance: {np.sum(mask)} / {len(eig_vals)}")
    
    eig_vals = eig_vals[mask]
    eig_vecs = eig_vecs[:, mask]
    
    # W = Lambda^(-1/2) V^T
    W = np.dot(eig_vecs, np.dot(np.diag(1.0 / np.sqrt(eig_vals)), eig_vecs.T))
    
    # 3. Whiten the Gain Matrix: G_tilde = W * G_proj
    G_tilde = np.dot(W, G_proj)
    
    # --- MNE Scaling Step ---
    n_nzero = np.sum(mask)
    trace_GRGT = np.sum(G_tilde ** 2)
    g_scale = np.sqrt(n_nzero / trace_GRGT)
    print(f"Gain scaling factor: {g_scale:.4e}")
    
    G_tilde = G_tilde * g_scale
    
    # 4. SVD of G_tilde = U * S * V^T
    U, S, Vh = scipy.linalg.svd(G_tilde, full_matrices=False)
    V = Vh.T
    
    # 5. Compute Inverse Operator Weights
    # Gamma = S / (S^2 + lambda2)
    S2 = S ** 2
    Gamma_diag = S / (S2 + lambda2)
    
    # 6. Assemble K
    # K = (V * Gamma) * U^T * W
    KV = V * Gamma_diag  # Broadcast
    KU = np.dot(U.T, W)
    K = np.dot(KV, KU)
    
    # Apply source covariance scaling
    K *= g_scale
    
    # 7. Apply K to data
    print(f"--- [Standalone] Applying {method} Inverse ---")
    source_estimate = np.dot(K, y)
    
    # 8. Apply dSPM normalization if requested
    if method == 'dSPM':
        # noise_var = diag( V * Gamma^2 * V^T ) * g_scale^2
        noise_var = np.sum((V * Gamma_diag) ** 2, axis=1) * (g_scale ** 2)
        noise_norm = 1.0 / np.sqrt(noise_var)
        source_estimate *= noise_norm[:, np.newaxis]
        
    return source_estimate


# -----------------------------------------------------------------------------
# 4. Evaluate Results
# -----------------------------------------------------------------------------
def evaluate_results(x_hat, info, evoked, forward, noise_cov):
    """
    Compares standalone results against the reference MNE implementation.
    Generates metrics and plots.
    """
    print("\n=== Phase 3: Reference MNE Reconstruction ===")
    
    # Create inverse operator using MNE (Reference)
    inv_mne = mne.minimum_norm.make_inverse_operator(
        info, forward, noise_cov, 
        loose=0.0, depth=None, fixed=True, verbose=False
    )
    
    # Apply MNE inverse
    stc_mne = mne.minimum_norm.apply_inverse(
        evoked, inv_mne, lambda2=1.0/9.0, method='dSPM', verbose=False
    )
    
    x_mne = stc_mne.data
    
    print("\n=== Phase 4: Evaluation ===")
    print(f"Standalone shape: {x_hat.shape}")
    print(f"MNE shape: {x_mne.shape}")
    
    # Compute Metrics
    mse = np.mean((x_hat - x_mne) ** 2)
    if mse == 0:
        psnr = np.inf
    else:
        psnr = 10 * np.log10(np.max(x_mne)**2 / mse)
    
    corr = np.corrcoef(x_hat.ravel(), x_mne.ravel())[0, 1]
    
    print(f"MSE between Standalone and MNE: {mse:.6e}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Correlation: {corr:.6f}")
    
    if corr > 0.99:
        print("SUCCESS: Standalone implementation matches MNE reference!")
    else:
        print("WARNING: Discrepancy detected.")
        
    # Visualization
    max_idx = np.argmax(np.sum(x_mne**2, axis=1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(evoked.times, x_mne[max_idx], label='MNE Reference', linewidth=2)
    plt.plot(evoked.times, x_hat[max_idx], '--', label='Standalone', linewidth=2)
    plt.title(f'Source Time Course (Vertex {max_idx})')
    plt.xlabel('Time (s)')
    plt.ylabel('dSPM value')
    plt.legend()
    plt.grid(True)
    output_img = 'comparison_plot.png'
    plt.savefig(output_img)
    print(f"Comparison plot saved to {output_img}")


if __name__ == '__main__':
    # 1. Load Data
    G, C, y, nave, P, info, evoked, fwd_obj, cov_obj = load_and_preprocess_data()
    
    # 2. Run Inversion (Standalone)
    x_hat = run_inversion(G, C, y, nave, P, snr=3.0, method='dSPM')
    
    # (Optional) Verify Forward Operator
    # We can check if projecting the estimated source back to sensor space 
    # roughly matches the data (though dSPM scales things, so magnitude will differ)
    # y_reconstructed = forward_operator(x_hat, G)
    
    # 3. Evaluate
    evaluate_results(x_hat, info, evoked, fwd_obj, cov_obj)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")