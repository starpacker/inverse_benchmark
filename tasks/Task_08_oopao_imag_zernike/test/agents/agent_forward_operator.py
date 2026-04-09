import numpy as np

def forward_operator(phase_map, tel):
    """
    Computes the PSF from the phase map using physical optics principles.
    PSF = | FFT( Amplitude * exp(i * Phase) ) |^2
    
    Args:
        phase_map: 2D array of phase values [radians]
        tel: Telescope object containing pupil information
        
    Returns:
        psf: 2D array of the Point Spread Function (normalized)
    """
    # 1. Get Pupil Amplitude (Binary mask)
    amplitude = tel.pupil
    
    # 2. Create Complex Field (Electric Field)
    # E = A * e^(i * phi)
    electric_field = amplitude * np.exp(1j * phase_map)
    
    # 3. Apply Zero Padding (for sampling)
    # A factor of 4 ensures Nyquist sampling of the intensity
    zero_padding = 4
    N = tel.resolution
    N_padded = N * zero_padding
    
    # Pad the electric field
    pad_width = (N_padded - N) // 2
    electric_field_padded = np.pad(electric_field, pad_width)
    
    # 4. Fourier Transform (Propagation to Focal Plane)
    # Shift before FFT to center zero frequency
    complex_focal_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(electric_field_padded)))
    
    # 5. Compute Intensity (PSF)
    psf = np.abs(complex_focal_plane)**2
    
    # Normalize
    psf = psf / psf.max()
    
    return psf