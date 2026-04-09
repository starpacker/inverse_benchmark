import numpy as np

import scipy.fft

def fft2c(x):
    """Centered 2D FFT."""
    return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(x)))

def ifft2c(x):
    """Centered 2D Inverse FFT."""
    return scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.ifftshift(x)))

def forward_operator(object_patch, probe):
    """
    Performs the physical forward model: Exit Wave -> FFT.
    
    Args:
        object_patch (np.array): Complex object patch.
        probe (np.array): Complex probe.
        
    Returns:
        np.array: The predicted far-field complex wave (before magnitude).
    """
    exit_wave = object_patch * probe
    wave_fourier = fft2c(exit_wave)
    return wave_fourier

def run_inversion(data_container, iterations=300, alpha=0.5):
    """
    Runs the ptychographic iterative engine.
    
    Args:
        data_container (dict): Data loaded from step 1.
        iterations (int): Number of epochs.
        alpha (float): Update step size parameter (beta in some literature).
        
    Returns:
        dict: Result containing 'reconstructed_object', 'reconstructed_probe', and 'errors'.
    """
    print("Starting Inversion (ePIE)...")
    
    # Unpack data
    ptychogram = data_container['ptychogram']
    positions = data_container['positions']
    No = data_container['No']
    Np = data_container['Np']
    
    # Mutable state
    obj_recon = data_container['initial_object'].copy()
    probe_recon = data_container['initial_probe'].copy()
    
    # Pre-calculate amplitude from intensity data
    measured_amplitudes = np.sqrt(ptychogram)
    num_pos = len(positions)
    error_history = []
    
    for i in range(iterations):
        err_sum = 0.0
        # Randomize order
        indices = np.random.permutation(num_pos)
        
        for idx in indices:
            r, c = positions[idx]
            
            # Boundary checks
            if r < 0 or c < 0 or r + Np > No or c + Np > No:
                continue
                
            # 1. Extract Object Patch
            obj_patch = obj_recon[r:r+Np, c:c+Np]
            
            # 2. Forward Propagation
            # Call the specific forward operator
            wave_fourier = forward_operator(obj_patch, probe_recon)
            
            # 3. Apply Modulus Constraint
            current_amp = np.abs(wave_fourier)
            # Avoid division by zero
            current_amp[current_amp < 1e-10] = 1e-10
            
            measured_amp = measured_amplitudes[idx]
            
            # Project to constraint manifold
            wave_fourier_updated = wave_fourier * (measured_amp / current_amp)
            
            # 4. Inverse Propagation
            exit_wave_updated = ifft2c(wave_fourier_updated)
            
            # Calculate difference for updates
            current_exit_wave = obj_patch * probe_recon
            diff = exit_wave_updated - current_exit_wave
            
            # Track error
            err_sum += np.sum(np.abs(diff)**2)
            
            # 5. Object Update (ePIE)
            absP2 = np.abs(probe_recon)**2
            Pmax = np.max(absP2)
            if Pmax < 1e-10: Pmax = 1e-10
            
            update_obj = alpha * (np.conj(probe_recon) * diff) / Pmax
            obj_recon[r:r+Np, c:c+Np] += update_obj
            
            # 6. Probe Update (ePIE)
            absO2 = np.abs(obj_patch)**2
            Omax = np.max(absO2)
            if Omax < 1e-10: Omax = 1e-10
            
            update_probe = alpha * (np.conj(obj_patch) * diff) / Omax
            probe_recon += update_probe
            
        print(f"  Iteration {i+1}/{iterations}, Error: {err_sum:.4e}")
        error_history.append(err_sum)
        
    return {
        'reconstructed_object': obj_recon,
        'reconstructed_probe': probe_recon,
        'error_history': error_history
    }
