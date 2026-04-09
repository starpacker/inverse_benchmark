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

def PS4(Inp0, Inp1, Inp2, Inp3):
    '''
    Function to recover the phase information of a sample from four DHM in-axis acquisitions holograms
    '''
    inp0 = np.array(Inp0)
    inp1 = np.array(Inp1)
    inp2 = np.array(Inp2)
    inp3 = np.array(Inp3)

    # compensation process
    # U_obj ~ (I3-I1)j + (I2-I0)
    comp_phase = (inp3 - inp1) * 1j + (inp2 - inp0)

    return comp_phase

def run_inversion(I_stack, z, wavelength, dx, dy):
    """
    Performs the reconstruction:
    1. Phase shifting retrieval (PS4).
    2. Back-propagation to object plane.
    
    Returns:
        reconstructed_field (np.complex): The complex object field at z=0.
    """
    I0, I1, I2, I3 = I_stack
    
    # Step 1: Recover Complex Field at Hologram Plane using PS4
    # The PS4 formula returns (I3-I1)j + (I2-I0) which is proportional to 4*R*O
    # Since R=1, we get 4*O. We need to divide by 4 to get true scale.
    recovered_holo_field = PS4(I0, I1, I2, I3) / 4.0
    
    # Step 2: Propagate back to the object plane (-z)
    reconstructed_field = angularSpectrum(recovered_holo_field, -z, wavelength, dx, dy)
    
    return reconstructed_field
