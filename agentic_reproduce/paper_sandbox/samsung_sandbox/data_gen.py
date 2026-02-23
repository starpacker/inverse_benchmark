import numpy as np
import os
from scipy import ndimage

def main():
    np.random.seed(42)
    os.makedirs("dataset", exist_ok=True)

    # ========== PARAMETERS ==========
    wavelength = 13.5e-9  # 13.5 nm
    pixel_size = 27e-9    # 27 nm
    n_pads = 15           # 15x15 pad array
    pad_pixels = 22       # pixels per pad
    duty_cycle = 0.6
    pitch_pixels = int(round(pad_pixels / duty_cycle))  # ~37 pixels
    
    # Optical constants at 13.5 nm (from CXRO)
    n_SiCN = complex(0.9791, 0.0059)
    n_SiO2 = complex(0.9780, 0.0108)
    n_Si = complex(0.9991, 0.0018)
    n_Cu = complex(0.9625, 0.0613)
    
    # Compute reflectivities (normal incidence, simplified Fresnel)
    # R = |(n-1)/(n+1)|^2 for normal incidence from vacuum
    def fresnel_r(n_mat):
        return (1 - n_mat) / (1 + n_mat)
    
    # Substrate reflectivity (simplified: SiCN on SiO2 on Si)
    r_substrate = fresnel_r(n_SiCN)
    R_substrate = np.abs(r_substrate)
    phase_substrate = np.angle(r_substrate)
    
    # Cu pad reflectivity
    r_cu = fresnel_r(n_Cu)
    R_cu = np.abs(r_cu)
    phase_cu = np.angle(r_cu)
    
    # ========== BUILD OBJECT ==========
    # Object grid size
    obj_size = n_pads * pitch_pixels + pad_pixels  # extra padding
    obj_size = max(obj_size, 600)  # ensure sufficient size
    
    # Create support mask S (binary: 1 inside pads, 0 outside)
    support = np.zeros((obj_size, obj_size), dtype=np.float64)
    
    # Assign random recession depths per pad: h in [1, 3] nm
    recession_depths = np.random.uniform(1e-9, 3e-9, size=(n_pads, n_pads))
    
    # Phase modulation: delta_phi = 4*pi*h / lambda
    phase_modulation = 4 * np.pi * recession_depths / wavelength
    
    # Build complex reflection function O(r)
    # Outside pads: substrate reflectivity Rp
    O = np.ones((obj_size, obj_size), dtype=np.complex128) * r_substrate
    
    # Place pads
    pad_centers = []
    offset = (obj_size - n_pads * pitch_pixels) // 2
    
    for i in range(n_pads):
        for j in range(n_pads):
            y_start = offset + i * pitch_pixels
            x_start = offset + j * pitch_pixels
            y_end = y_start + pad_pixels
            x_end = x_start + pad_pixels
            
            if y_end <= obj_size and x_end <= obj_size:
                support[y_start:y_end, x_start:x_end] = 1.0
                
                # Cu pad reflection with recession depth phase
                h = recession_depths[i, j]
                delta_phi = 4 * np.pi * h / wavelength
                # Pad reflectivity: Cu reflectivity * phase from recession
                O[y_start:y_end, x_start:x_end] = r_cu * np.exp(1j * delta_phi)
                
                pad_centers.append((y_start + pad_pixels // 2, x_start + pad_pixels // 2))
    
    # Ground truth: complex object O(r)
    # We store amplitude and phase as a 2-channel real array
    gt_amplitude = np.abs(O).astype(np.float32)
    gt_phase = np.angle(O).astype(np.float32)
    
    # For evaluation, the key quantity is the phase map
    # gt_output: complex object stored as (2, H, W) - [amplitude, phase]
    gt_output = np.stack([gt_amplitude, gt_phase], axis=0).astype(np.float32)
    
    # ========== PROBE FUNCTION (Airy spot) ==========
    probe_size = 256  # probe window size
    
    # Create Airy spot probe
    y_probe = np.arange(probe_size) - probe_size // 2
    x_probe = np.arange(probe_size) - probe_size // 2
    YP, XP = np.meshgrid(y_probe, x_probe, indexing='ij')
    r_probe = np.sqrt(XP**2 + YP**2) * pixel_size
    
    # Airy pattern: J1(x)/x, approximate with sinc-like function
    # NA chosen so that probe covers several pads for good overlap
    NA = 0.08
    k_val = 2 * np.pi / wavelength
    arg = k_val * NA * r_probe
    
    # Airy function: 2*J1(x)/x
    from scipy.special import j1
    with np.errstate(divide='ignore', invalid='ignore'):
        airy = np.where(arg == 0, 1.0, 2 * j1(arg) / arg)
    
    probe = airy.astype(np.complex128)
    probe = probe / np.max(np.abs(probe))  # normalize
    
    # ========== PTYCHOGRAPHIC SCANNING ==========
    step_size = 40  # pixels, ~75% overlap with probe
    
    # Scan positions
    scan_range_y = range(0, obj_size - probe_size, step_size)
    scan_range_x = range(0, obj_size - probe_size, step_size)
    
    scan_positions = []
    for sy in scan_range_y:
        for sx in scan_range_x:
            scan_positions.append((sy, sx))
    
    n_positions = len(scan_positions)
    
    # ========== FORWARD MODEL: Generate diffraction patterns ==========
    detector_size = probe_size  # same as probe window
    
    diffraction_patterns = np.zeros((n_positions, detector_size, detector_size), dtype=np.float64)
    
    noise_level = 0.01  # Gaussian noise relative level
    
    for idx, (sy, sx) in enumerate(scan_positions):
        # Extract object patch
        obj_patch = O[sy:sy+probe_size, sx:sx+probe_size]
        
        # Exit wave
        exit_wave = probe * obj_patch
        
        # Far-field diffraction (Fraunhofer): FFT
        diff_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_wave)))
        
        # Intensity
        intensity = np.abs(diff_field)**2
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * np.mean(intensity), intensity.shape)
        intensity_noisy = np.maximum(intensity + noise, 0)
        
        diffraction_patterns[idx] = intensity_noisy
    
    # Store scan positions as well (needed for reconstruction)
    scan_pos_array = np.array(scan_positions, dtype=np.float32)
    
    # Input: diffraction patterns + scan positions + probe + support
    # We pack essential info. For simplicity, save diffraction patterns as input
    # and auxiliary data separately
    
    # To keep it manageable, limit number of scan positions if too many
    max_positions = min(n_positions, 400)
    if n_positions > max_positions:
        indices = np.random.choice(n_positions, max_positions, replace=False)
        indices.sort()
        diffraction_patterns = diffraction_patterns[indices]
        scan_pos_array = scan_pos_array[indices]
        n_positions = max_positions
    
    # ========== BASELINE: Gerchberg-Saxton / Error Reduction with reflectivity prior ==========
    # Standard phase retrieval baseline using averaged diffraction pattern
    # This is a simplified single-pattern phase retrieval (error reduction algorithm)
    
    # Use the central diffraction pattern for single-shot reconstruction
    center_idx = n_positions // 2
    center_pos = scan_pos_array[center_idx].astype(int)
    center_dp = diffraction_patterns[center_idx]
    measured_amplitude = np.sqrt(center_dp)
    
    # Error Reduction (Gerchberg-Saxton) algorithm
    n_iterations = 300
    
    # Initialize with random phase
    np.random.seed(123)
    current_phase = np.random.uniform(-np.pi, np.pi, (detector_size, detector_size))
    
    # Support for this patch
    sy_c, sx_c = int(center_pos[0]), int(center_pos[1])
    support_patch = support[sy_c:sy_c+probe_size, sx_c:sx_c+probe_size]
    probe_abs = np.abs(probe)
    
    # Known substrate reflectivity (complex)
    Rp = r_substrate
    
    beta = 0.9  # feedback parameter
    
    # Reconstruct exit wave, then extract object
    current_estimate = measured_amplitude * np.exp(1j * current_phase)
    
    for iteration in range(n_iterations):
        # Inverse FFT to real space
        real_space = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(current_estimate)))
        
        # Apply constraints in real space
        # Inside support: keep the estimate (divide by probe to get object)
        # Outside support: enforce substrate reflectivity
        
        # Object estimate
        with np.errstate(divide='ignore', invalid='ignore'):
            obj_est = np.where(probe_abs > 1e-10, 
                              real_space / (probe + 1e-10), 
                              Rp)
        
        # Apply support constraint with reflectivity prior (rpIO-style, Eq. 5)
        obj_constrained = np.where(support_patch > 0.5,
                                    obj_est,
                                    beta * obj_est + (1 - beta) * Rp)
        
        # Back to exit wave
        exit_wave_constrained = probe * obj_constrained
        
        # FFT to detector
        detector_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_wave_constrained)))
        
        # Replace amplitude with measured, keep phase
        det_phase = np.angle(detector_field)
        current_estimate = measured_amplitude * np.exp(1j * det_phase)
    
    # Final real-space estimate
    real_space_final = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(current_estimate)))
    with np.errstate(divide='ignore', invalid='ignore'):
        obj_baseline_patch = np.where(probe_abs > 1e-10,
                                       real_space_final / (probe + 1e-10),
                                       Rp)
    
    # ========== FULL rpPIE BASELINE ==========
    # Implement simplified rpPIE (Eq. 6) as the classical baseline
    
    # Initialize object estimate
    O_est = np.ones((obj_size, obj_size), dtype=np.complex128) * Rp
    
    alpha_reg = 1e-3  # regularization
    beta_rpie = 0.9
    n_rpie_iterations = 50
    
    P_max = np.max(np.abs(probe))
    
    for iteration in range(n_rpie_iterations):
        # Shuffle scan positions
        order = np.random.permutation(n_positions)
        
        for pos_idx in order:
            sy_p, sx_p = int(scan_pos_array[pos_idx, 0]), int(scan_pos_array[pos_idx, 1])
            
            # Extract object patch
            obj_patch = O_est[sy_p:sy_p+probe_size, sx_p:sx_p+probe_size].copy()
            
            # Exit wave
            psi = probe * obj_patch
            
            # Propagate to detector
            Psi = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psi)))
            
            # Measured amplitude
            meas_amp = np.sqrt(diffraction_patterns[pos_idx])
            
            # Replace amplitude
            Psi_phase = np.angle(Psi)
            Psi_corrected = meas_amp * np.exp(1j * Psi_phase)
            
            # Back-propagate
            psi_corrected = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Psi_corrected)))
            
            # ePIE update (Step 1 of Eq. 6)
            probe_conj = np.conj(probe)
            update_weight = (np.abs(probe) / P_max) * probe_conj / (np.abs(probe)**2 + alpha_reg)
            
            obj_patch_updated = obj_patch + beta_rpie * update_weight * (psi_corrected - psi)
            
            # Step 2: enforce reflectivity prior outside support
            support_patch_local = support[sy_p:sy_p+probe_size, sx_p:sx_p+probe_size]
            obj_patch_updated = np.where(
                support_patch_local > 0.5,
                obj_patch_updated,
                beta_rpie * obj_patch_updated + (1 - beta_rpie) * Rp
            )
            
            # Write back
            O_est[sy_p:sy_p+probe_size, sx_p:sx_p+probe_size] = obj_patch_updated
    
    # Baseline output
    baseline_amplitude = np.abs(O_est).astype(np.float32)
    baseline_phase = np.angle(O_est).astype(np.float32)
    baseline_output = np.stack([baseline_amplitude, baseline_phase], axis=0).astype(np.float32)
    
    # ========== PACK INPUT DATA ==========
    # Input contains: diffraction patterns (primary measurement)
    # We also save auxiliary data needed for reconstruction
    
    # Save everything
    np.save("dataset/gt_output.npy", gt_output)
    
    # For input, save diffraction patterns (the actual measurements)
    # Also save metadata as separate files for completeness
    input_data = diffraction_patterns.astype(np.float32)
    np.save("dataset/input.npy", input_data)
    
    np.save("dataset/baseline.npy", baseline_output)
    
    # Save auxiliary data for reconstruction algorithms
    np.save("dataset/scan_positions.npy", scan_pos_array)
    np.save("dataset/probe.npy", np.stack([np.real(probe), np.imag(probe)], axis=0).astype(np.float32))
    np.save("dataset/support.npy", support.astype(np.float32))
    
    # ========== PRINT DIAGNOSTICS ==========
    print(f"Object size: {obj_size}x{obj_size}")
    print(f"Number of scan positions: {n_positions}")
    print(f"Probe size: {probe_size}x{probe_size}")
    print(f"Diffraction patterns shape: {input_data.shape}")
    print(f"GT output shape: {gt_output.shape}")
    print(f"Baseline output shape: {baseline_output.shape}")
    
    # Compute metrics
    # Phase RMSE on pad regions
    gt_phase_map = gt_output[1]
    bl_phase_map = baseline_output[1]
    
    # Crop to same region for comparison
    pad_mask = support.astype(bool)
    
    if pad_mask.any():
        gt_phase_pads = gt_phase_map[pad_mask]
        bl_phase_pads = bl_phase_map[pad_mask]
        
        # Remove global phase offset
        phase_diff = bl_phase_pads - gt_phase_pads
        offset = np.mean(phase_diff)
        bl_phase_pads_aligned = bl_phase_pads - offset
        
        rmse = np.sqrt(np.mean((bl_phase_pads_aligned - gt_phase_pads)**2))
        print(f"Phase RMSE on pads (baseline): {rmse:.4f} rad")
    
    # Amplitude comparison
    gt_amp_map = gt_output[0]
    bl_amp_map = baseline_output[0]
    
    amp_range = np.max(gt_amp_map) - np.min(gt_amp_map)
    if amp_range > 0:
        mse_amp = np.mean((gt_amp_map - bl_amp_map)**2)
        psnr_amp = 10 * np.log10(amp_range**2 / (mse_amp + 1e-10))
        print(f"Amplitude PSNR (baseline): {psnr_amp:.2f} dB")
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()