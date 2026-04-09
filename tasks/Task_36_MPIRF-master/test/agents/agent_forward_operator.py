import numpy as np

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
