import numpy as np

def forward_operator(raw_input, settings):
    """
    Represents the physical transform from Raw OCT Fringes to Tomogram (Complex Space).
    In this specific pipeline, 'forward' implies processing the raw interference pattern 
    into the spatial domain (Inverse Scattering), as this is the primary computational step.
    
    Steps: FFT -> Zoom/Shift -> IFFT -> Resample (Dechirp) -> Dispersion Comp -> Window -> FFT -> Shift
    """
    ch1, ch2 = raw_input
    
    num_samples = settings['numSamples']
    fourier_len = num_samples // 2
    zoom = settings['zoomFactor']
    zoomed_len = num_samples * zoom
    num_z_out = settings['numZOut']
    
    # 1. FFT
    ft1 = np.fft.fft(ch1, axis=0)
    ft2 = np.fft.fft(ch2, axis=0)
    
    # 2. Prepare Zoom Array (Pad in frequency domain)
    zoom_ch1 = np.zeros((zoomed_len, ch1.shape[1]), dtype='complex64')
    zoom_ch2 = np.zeros((zoomed_len, ch2.shape[1]), dtype='complex64')
    
    zoom_ch1[:fourier_len, :] = ft1[:fourier_len, :]
    zoom_ch2[:fourier_len, :] = ft2[:fourier_len, :]
    
    # 3. Demodulation Shift (Pre-IFFT)
    demod_idx = settings['demod_idx']
    zoom_ch1 = np.roll(zoom_ch1, -demod_idx, axis=0)
    zoom_ch2 = np.roll(zoom_ch2, -demod_idx, axis=0)
    
    # 4. IFFT (to high-res time/fringe domain)
    zoom_ch1 = zoom * num_samples * np.fft.ifft(zoom_ch1, axis=0)
    zoom_ch2 = zoom * num_samples * np.fft.ifft(zoom_ch2, axis=0)
    
    # 5. Non-uniform Resampling (Dechirp)
    chirp_indices = settings['chirp_indices']
    k1 = zoom_ch1[chirp_indices, :]
    k2 = zoom_ch2[chirp_indices, :]
    
    # 6. Dispersion Compensation
    disp_vec = settings['dispersion_vector'][:, None]
    k1 = k1 * disp_vec
    k2 = k2 * disp_vec
    
    # 7. Windowing
    window = settings['fringe_window']
    k1 = k1 * window
    k2 = k2 * window
    
    # 8. Final FFT (k-space to z-space)
    tom1 = np.fft.fft(k1, n=num_z_out, axis=0) * 1e-6
    tom2 = np.fft.fft(k2, n=num_z_out, axis=0) * 1e-6
    
    # 9. Final Demodulation Shift (Post-FFT)
    demod_rev_idx = settings['demod_rev_idx']
    tom1 = np.roll(tom1, demod_rev_idx, axis=0)
    tom2 = np.roll(tom2, demod_rev_idx, axis=0)
    
    # 10. Flip
    if settings['flipUpDown']:
        tom1 = np.flipud(tom1)
        tom2 = np.flipud(tom2)
        
    return tom1, tom2
