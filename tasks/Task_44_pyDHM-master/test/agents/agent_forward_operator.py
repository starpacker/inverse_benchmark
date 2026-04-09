import numpy as np

from math import pi, sqrt, log10

def angularSpectrum(field, z, wavelength, dx, dy):
    """
    Function to diffract a complex field using the angular spectrum approximation
    Extracted logic from input code.
    """
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * N)
    dfy = 1 / (dy * M)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    # Transfer function
    # Note: Using np.exp for phase
    phase_term = np.exp(1j * z * 2 * pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2)) + 0j))

    tmp = field_spec * phase_term

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out

def forward_operator(field_input, z, wavelength, dx, dy, add_quantization=True):
    """
    Simulates the hologram recording process:
    1. Propagation to hologram plane.
    2. Interference with phase-shifted reference waves.
    3. (Optional) Camera quantization.
    
    Returns:
        I_stack (list of np.array): List containing [I0, I1, I2, I3].
    """
    # Propagate the field to the hologram plane
    hologram_field_complex = angularSpectrum(field_input, z, wavelength, dx, dy)
    
    # Simulate Reference Wave (Plane Wave, Amplitude 1)
    R = 1.0 
    
    # Phase shifts: 0, pi/2, pi, 3pi/2
    I0 = np.abs(hologram_field_complex + R * np.exp(1j * 0))**2
    I1 = np.abs(hologram_field_complex + R * np.exp(1j * pi/2))**2
    I2 = np.abs(hologram_field_complex + R * np.exp(1j * pi))**2
    I3 = np.abs(hologram_field_complex + R * np.exp(1j * 3*pi/2))**2
    
    if add_quantization:
        # Simulate 8-bit Camera Quantization
        # Normalize by the global maximum of the stack to preserve relative intensities
        max_val = np.max([I0.max(), I1.max(), I2.max(), I3.max()])
        
        def quantize_single(img, limit):
            img_norm = img / limit
            img_8bit = np.round(img_norm * 255).astype(np.uint8)
            # Convert back to float scale
            return img_8bit.astype(np.float32) / 255.0 * limit

        I0 = quantize_single(I0, max_val)
        I1 = quantize_single(I1, max_val)
        I2 = quantize_single(I2, max_val)
        I3 = quantize_single(I3, max_val)
        
    return [I0, I1, I2, I3]
