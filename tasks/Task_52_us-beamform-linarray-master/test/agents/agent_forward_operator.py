import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def forward_operator(data_preproc, time_vec, probe_coords, params, method='dynamic'):
    """
    Performs beamforming (The core operator).
    
    Args:
        data_preproc: (N_beams, N_channels, Time)
        time_vec: Time array corresponding to data
        probe_coords: X coordinates of probe elements
        params: Dictionary of constants
        method: 'dynamic' or 'fixed'
    
    Returns:
        image_lines: (N_beams, Depth_samples) raw RF beamformed data
    """
    speed_sound = params['SPEED_SOUND']
    n_probe_channels = params['N_PROBE_CHANNELS']
    n_transmit_beams = params['N_TRANSMIT_BEAMS']
    
    fs = 1 / (time_vec[1] - time_vec[0])
    
    if method == 'fixed':
        # Fixed Focus Beamforming
        receive_focus = 30e-3 # Fixed focal depth
        
        # Calculate delays for fixed focus
        delay_ind = np.zeros(n_probe_channels, dtype=int)
        for r in range(n_probe_channels):
            delay = receive_focus / speed_sound * (np.sqrt((probe_coords[r] / receive_focus)**2 + 1) - 1)
            delay_ind[r] = int(round(delay * fs))
        max_delay = np.max(delay_ind)

        waveform_length = data_preproc.shape[2]
        image_lines = np.zeros((n_transmit_beams, waveform_length))
        
        for q in range(n_transmit_beams):
            scan_line = np.zeros(waveform_length + max_delay)
            for r in range(n_probe_channels):
                pad_len = delay_ind[r]
                waveform = data_preproc[q, r, :]
                
                # Construct padded array manually to match shapes
                # padding at end to match scan_line length: (waveform_length + max_delay) - waveform_len - pad_len
                total_len = waveform_length + max_delay
                remaining = total_len - len(waveform) - pad_len
                
                # Logic from original code: concat(fill_pad, waveform, delay_pad)
                # Original: fill_pad = zeros(len(scan_line) - waveform_length - delay_ind[r])
                # The original logic shifted signal. 
                # Let's reproduce:
                fill_pad = np.zeros(remaining) 
                delay_pad = np.zeros(pad_len)
                
                # Note: Original code logic: scan_line = scan_line + np.concatenate((fill_pad, waveform, delay_pad))
                # This essentially shifts the waveform in the buffer.
                shifted_waveform = np.concatenate((fill_pad, waveform, delay_pad))
                
                # Safety check for size
                if len(shifted_waveform) != len(scan_line):
                     # Adjust if slight rounding error
                     diff = len(scan_line) - len(shifted_waveform)
                     if diff > 0:
                         shifted_waveform = np.concatenate((shifted_waveform, np.zeros(diff)))
                     elif diff < 0:
                         shifted_waveform = shifted_waveform[:diff]

                scan_line = scan_line + shifted_waveform
            
            image_lines[q, :] = scan_line[max_delay:]
            
    elif method == 'dynamic':
        # Dynamic Focusing Beamforming
        zd = time_vec * speed_sound / 2
        zd2 = zd**2
        
        # Precompute indices
        prop_dist = np.zeros((n_probe_channels, len(zd)))
        for r in range(n_probe_channels):
            dist1 = zd # Receive distance (approximation)
            dist2 = np.sqrt(probe_coords[r]**2 + zd2)
            prop_dist[r, :] = dist1 + dist2

        prop_dist_ind = np.round(prop_dist / speed_sound * fs).astype('int')
        
        # Handle out of bounds
        max_idx = len(time_vec) - 1
        prop_dist_ind[prop_dist_ind > max_idx] = max_idx
        prop_dist_ind[prop_dist_ind < 0] = 0

        image_lines = np.zeros((n_transmit_beams, len(zd)))
        
        for q in range(n_transmit_beams):
            data_received = data_preproc[q, ...]
            scan_line = np.zeros(len(zd))
            for r in range(n_probe_channels):
                v = data_received[r, :]
                # Vectorized indexing
                scan_line = scan_line + v[prop_dist_ind[r, :]]
            image_lines[q, :] = scan_line
            
    else:
        raise ValueError(f"Unknown beamforming method: {method}")
        
    return image_lines
