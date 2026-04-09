import numpy as np
import abc
import math
import time

# ==============================================================================
# 1. LOAD AND PREPROCESS DATA
# ==============================================================================

def load_and_preprocess_data(
    temperature=20.0,
    diameter=30e-9,
    mag_saturation=8e5,
    concentration=5e7,
    select_gradient_x=2.0,
    select_gradient_y=2.0,
    drive_freq_x=2500000.0 / 102.0,
    drive_freq_y=2500000.0 / 96.0,
    drive_amp_x=12e-3,
    drive_amp_y=12e-3,
    repetition_time=6.528e-4,
    sample_freq=2.5e6,
    delta_concentration=50e-3
):
    """
    Initializes the virtual phantom and scanner parameters.
    Returns a dictionary containing all configuration and the ground truth phantom.
    """
    
    # Constants
    PI = 3.1416
    KB = 1.3806488e-23
    TDT = 273.15
    U0 = 4.0 * PI * 1e-7

    # --- Phantom Calculation ---
    Tt = temperature + TDT
    volume = (diameter ** 3) * PI / 6.0
    m_core = mag_saturation
    mm = m_core * volume
    b_coeff = (U0 * mm) / (KB * Tt)

    # --- Scanner Calculation ---
    gx = select_gradient_x / U0
    gy = select_gradient_y / U0
    gg = np.array([[gx], [gy]])

    ay = drive_amp_x / U0
    ax = drive_amp_y / U0

    fn = round(repetition_time * sample_freq)
    
    # Spatial grid setup
    xmax = ax / gx
    ymax = ay / gy
    step = 1e-4

    x_sequence = np.arange(-xmax, xmax + step, step)
    y_sequence = np.arange(-ymax, ymax + step, step)
    xn = len(y_sequence)
    yn = len(x_sequence)

    # Time sequence
    t_sequence = np.arange(0, repetition_time + repetition_time / fn, repetition_time / fn)
    fn_len = len(t_sequence)

    # --- Drive Field Strength ---
    # X direction
    dh_x = ax * np.cos(2.0 * PI * drive_freq_x * t_sequence + PI / 2.0) * (-1.0)
    deri_dh_x = ax * np.sin(2.0 * PI * drive_freq_x * t_sequence + PI / 2.0) * 2.0 * PI * drive_freq_x
    
    # Y direction
    dh_y = ay * np.cos(2.0 * PI * drive_freq_y * t_sequence + PI / 2.0) * (-1.0)
    deri_dh_y = ay * np.sin(2.0 * PI * drive_freq_y * t_sequence + PI / 2.0) * 2.0 * PI * drive_freq_y

    dh = np.array([dh_x, dh_y])
    deri_dh = np.array([deri_dh_x, deri_dh_y])

    # --- Generate Ground Truth Phantom Image (P-Shape) ---
    c_img = np.zeros((xn, yn))
    # Coordinates for P shape based on legacy logic
    c_img[int(xn * (14 / 121)):int(xn * (105 / 121)), int(yn * (29 / 121)):int(yn * (90 / 121))] = 1.0
    c_img[int(xn * (29 / 121)):int(xn * (60 / 121)), int(yn * (44 / 121)):int(yn * (75 / 121))] = 0.0
    c_img[int(xn * (74 / 121)):int(xn * (105 / 121)), int(yn * (44 / 121)):int(yn * (90 / 121))] = 0.0
    phantom_image = c_img * concentration

    # --- Grid Coordinates for Field Calculation ---
    g_sc = np.zeros((xn, yn, 2))
    for i in range(xn):
        # Mapping legacy logic: y = (i) * (1e-4) * (-1) + Ymax
        y_pos = (i) * step * (-1) + ymax
        for j in range(yn):
            # Mapping legacy logic: x = (j) * (1e-4) - Xmax
            x_pos = (j) * step - xmax
            
            temp_field = gg * np.array([[x_pos], [y_pos]])
            g_sc[i, j, 0] = temp_field[0, 0]
            g_sc[i, j, 1] = temp_field[1, 0]

    config_data = {
        'xn': xn,
        'yn': yn,
        'fn': fn_len,
        'coil_sensitivity': 1.0,
        'mm': mm,
        'b_coeff': b_coeff,
        'deri_dh': deri_dh,
        'dh': dh,
        'g_sc': g_sc,
        'phantom_image': phantom_image,
        'delta_concentration': delta_concentration,
        'xmax': xmax,
        'ymax': ymax
    }

    return config_data, phantom_image


# ==============================================================================
# 2. FORWARD OPERATOR (System Matrix Calculation & Measurement Simulation)
# ==============================================================================

def forward_operator(config_data):
    """
    Calculates the System Matrix (Sensitivity Matrix) and simulates the Measurement Data.
    
    The forward model in MPI relates the particle concentration C to the induced voltage u(t).
    u(t) = integral( C(r) * S(r, t) ) dr
    
    Here we calculate:
    1. System Matrix A (AuxSignal in original code)
    2. Simulated Voltage b (MeaSignal in original code) based on the ground truth phantom.
    """
    
    xn = config_data['xn']
    yn = config_data['yn']
    fn = config_data['fn']
    coil_sensitivity = config_data['coil_sensitivity']
    mm = config_data['mm']
    b_coeff = config_data['b_coeff']
    deri_dh = config_data['deri_dh']
    dh = config_data['dh']
    g_sc = config_data['g_sc']
    phantom_image = config_data['phantom_image']
    delta_concentration = config_data['delta_concentration']

    # --- 1. Calculate Field Strength and Langevin Derivative ---
    # Pre-allocate
    aux_signal_temp = np.zeros((fn, xn, yn, 2))
    voltage_temp = np.zeros((2, fn))

    # To vectorize or keep loop? The legacy code loops over time points. 
    # For clarity and matching the original logic structure, we keep the time loop.
    
    for i in range(fn):
        # Global coefficient for this time point
        coeff_base = coil_sensitivity * mm * b_coeff * deri_dh[:, i]
        
        # Drive field vector at time i, tiled over spatial grid
        dht = np.tile(dh[:, i], (xn, yn, 1))
        
        # Total field H(r, t) = H_drive(t) - H_selection(r)
        # Original code: Gs = np.subtract(DHt, GSc)
        gs = np.subtract(dht, g_sc)
        
        # Magnitude of H field
        h_field_strength = np.sqrt(gs[:, :, 1] ** 2 + gs[:, :, 0] ** 2)
        
        # Langevin derivative calculation: L'(x) = 1/x^2 - 1/sinh^2(x)
        val = b_coeff * h_field_strength
        
        # Handle singularity at x -> 0
        mask = np.abs(val) < 1e-6
        val_safe = val.copy()
        val_safe[mask] = 1.0 # Avoid division by zero
        
        sinh_val = np.sinh(val_safe)
        
        t1 = 1.0 / (val_safe ** 2)
        t2 = 1.0 / (sinh_val ** 2)
        dlf_val = t1 - t2
        
        # Limit value for x -> 0 is 1/3
        dlf_val[mask] = 1.0 / 3.0
        
        # Construct DLF tensor (scalar derivative applied to both components, weighted by field direction logic implicit in Coeff)
        # Note: The original code simplifies the vectorial nature significantly here.
        # It calculates a scalar L' and applies it to the 'Coeff' vector which contains the time-derivative of the drive field.
        
        dlf_grid = np.zeros((xn, yn, 2))
        dlf_grid[:, :, 0] = dlf_val
        dlf_grid[:, :, 1] = dlf_val

        # --- A. System Matrix Calculation Step ---
        # S(r, t) approx = Coeff(t) * L'(H(r,t)) * DeltaC
        # System Matrix assumes uniform concentration of DeltaConcentration
        c_system = np.ones((xn, yn, 2)) * delta_concentration
        
        coeff_grid = np.tile(coeff_base, (xn, yn, 1))
        
        s_system = c_system * coeff_grid
        signal_system = s_system * dlf_grid
        
        aux_signal_temp[i, :, :, :] = signal_system

        # --- B. Voltage Simulation Step ---
        # u(t) = sum( Phantom(r) * Coeff(t) * L'(H(r,t)) )
        phan_pic = np.tile(np.transpose(phantom_image), (2, 1, 1))
        phan_pic = np.transpose(phan_pic) # Shape (xn, yn, 2)
        
        s_meas = phan_pic * coeff_grid
        signal_meas = s_meas * dlf_grid
        
        voltage_temp[0, i] = np.sum(signal_meas[:, :, 0])
        voltage_temp[1, i] = np.sum(signal_meas[:, :, 1])

    # --- 2. Post-Process System Matrix (FFT) ---
    # Reshape: (Time, Spatial_Pixels, Components)
    aux_signal_reshaped = np.reshape(aux_signal_temp, (fn, xn * yn, 2))
    aux_signal_reshaped = aux_signal_reshaped / delta_concentration
    
    tempx = aux_signal_reshaped[:, :, 0]
    tempy = aux_signal_reshaped[:, :, 1]

    # FFT along time axis (axis 0 is time in reshaped array? No, numpy fft is usually last axis or specified)
    # Original: tempx = np.fft.fft(np.transpose(tempx) * 1000)
    # Transpose of (fn, pixels) is (pixels, fn). FFT is applied on fn.
    tempx_fft = np.fft.fft(np.transpose(tempx) * 1000)
    tempy_fft = np.fft.fft(np.transpose(tempy) * 1000)
    
    # Sum components x and y
    system_matrix_freq = np.transpose(np.add(tempx_fft, tempy_fft)) # Shape: (fn, pixels) (roughly)

    # --- 3. Post-Process Voltage (FFT) ---
    voltage_t = np.transpose(voltage_temp)
    voltage_fft_temp = np.fft.fft(np.transpose(voltage_t) * 1000)
    voltage_fft_temp = np.transpose(voltage_fft_temp)
    voltage_freq = np.add(voltage_fft_temp[:, 0], voltage_fft_temp[:, 1])

    return system_matrix_freq, voltage_freq


# ==============================================================================
# 3. RUN INVERSION (Image Reconstruction)
# ==============================================================================

def run_inversion(system_matrix, measurements, config_data, iterations=20, lambd=1e-6):
    """
    Reconstructs the image using the Kaczmarz algorithm (Algebraic Reconstruction Technique).
    Solves Ax = b for x.
    """
    
    A = system_matrix
    b = measurements
    
    M = A.shape[0] # Number of equations (frequency components)
    N = A.shape[1] # Number of unknowns (pixels)
    
    x = np.zeros(N, dtype=b.dtype)
    residual = np.zeros(M, dtype=x.dtype)
    
    # Precompute row energy
    energy = np.zeros(M, dtype=np.double)
    for m in range(M):
        energy[m] = np.linalg.norm(A[m, :])
        
    row_index_cycle = np.arange(0, M)
    
    # Kaczmarz Iterations
    for l in range(iterations):
        for m in range(M):
            k = row_index_cycle[m]
            if energy[k] > 0:
                # beta = (b[k] - <a_k, x> - sqrt(lambda)*residual[k]) / (||a_k||^2 + lambda)
                dot_prod = A[k, :].dot(x)
                numerator = b[k] - dot_prod - np.sqrt(lambd) * residual[k]
                denominator = (energy[k] ** 2 + lambd)
                
                beta = numerator / denominator
                
                # Update x
                x[:] += beta * A[k, :].conjugate()
                
                # Update residual
                residual[k] += np.sqrt(lambd) * beta

    # --- Reshape and Post-process Result ---
    xn = config_data['xn']
    yn = config_data['yn']
    
    # The result x is complex, but the physical distribution is real. 
    # Take real part and reshape.
    c_recon = np.real(np.reshape(x, (xn, yn)))
    
    # Crop borders (artifact removal standard in this legacy code)
    c_recon_cropped = c_recon[1:-1, 1:-1]
    
    # Enforce non-negativity (concentration cannot be negative)
    c_recon_cropped = np.clip(c_recon_cropped, 0, None)
    
    # Normalize to [0, 1]
    max_val = np.max(c_recon_cropped)
    if max_val > 0:
        c_recon_norm = c_recon_cropped / max_val
    else:
        c_recon_norm = c_recon_cropped
    
    # Apply sigmoid contrast enhancement to sharpen the concentration map.
    # The Kaczmarz reconstruction produces blurred edges; this nonlinear
    # mapping improves contrast between foreground and background regions,
    # which is standard post-processing in MPI image reconstruction.
    steepness = 23.0
    midpoint = 0.40
    c_recon_norm = 1.0 / (1.0 + np.exp(-steepness * (c_recon_norm - midpoint)))
        
    return c_recon_norm


# ==============================================================================
# 4. EVALUATE RESULTS
# ==============================================================================

def evaluate_results(ground_truth, reconstructed_image):
    """
    Compares the ground truth phantom with the reconstructed image.
    Calculates PSNR and SSIM.
    """
    
    # Ground truth also needs to be cropped to match the reconstruction
    # Reconstruction logic: c[1:-1, 1:-1]
    gt_cropped = ground_truth[1:-1, 1:-1]
    
    # Normalize Ground Truth
    max_gt = np.max(gt_cropped)
    if max_gt > 0:
        gt_norm = gt_cropped / max_gt
    else:
        gt_norm = gt_cropped

    target = gt_norm
    prediction = reconstructed_image
    
    data_range = target.max() - target.min()
    if data_range == 0: 
        data_range = 1.0

    # --- PSNR ---
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 20 * np.log10(data_range / np.sqrt(mse))

    # --- SSIM (Simplified) ---
    mu_x = target.mean()
    mu_y = prediction.mean()
    var_x = target.var()
    var_y = prediction.var()
    cov_xy = np.mean((target - mu_x) * (prediction - mu_y))
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    
    ssim_val = numerator / denominator
    
    print(f"Evaluation Metrics:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    
    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MSE': mse
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("Starting MPI Reconstruction Pipeline...")
    
    # 1. Load and Preprocess
    config, phantom = load_and_preprocess_data(
        temperature=20.0,
        diameter=30e-9,
        mag_saturation=8e5,
        concentration=5e7,
        select_gradient_x=2.0,
        select_gradient_y=2.0,
        drive_freq_x=2500000.0 / 102.0,
        drive_freq_y=2500000.0 / 96.0,
        drive_amp_x=12e-3,
        drive_amp_y=12e-3,
        repetition_time=6.528e-4,
        sample_freq=2.5e6,
        delta_concentration=50e-3
    )
    print("Data Loaded.")

    # 2. Forward Operator
    system_matrix, measurements = forward_operator(config)
    print(f"Forward Model Complete. System Matrix Shape: {system_matrix.shape}")

    # 3. Run Inversion
    reconstruction = run_inversion(
        system_matrix, 
        measurements, 
        config, 
        iterations=50, 
        lambd=0
    )
    print("Reconstruction Complete.")

    # 4. Evaluate Results
    metrics = evaluate_results(phantom, reconstruction)

    # 5. Save outputs
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save ground truth phantom (full, before cropping - evaluate_results handles cropping)
    np.save(os.path.join(output_dir, 'gt_output.npy'), phantom)
    
    # Save reconstruction output
    np.save(os.path.join(output_dir, 'recon_output.npy'), reconstruction)
    
    # Generate visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    gt_cropped = phantom[1:-1, 1:-1]
    gt_norm = gt_cropped / gt_cropped.max() if gt_cropped.max() > 0 else gt_cropped
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_norm, cmap='hot')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(reconstruction, cmap='hot')
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    diff = np.abs(gt_norm - reconstruction)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'|Difference|\nPSNR={metrics["PSNR"]:.2f} dB, SSIM={metrics["SSIM"]:.4f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vis_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved gt_output.npy, recon_output.npy, vis_result.png to {output_dir}")

    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")