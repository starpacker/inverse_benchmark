import numpy as np

import scipy.linalg

import os

import mne

from mne.datasets import sample

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
