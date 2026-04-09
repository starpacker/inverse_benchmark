import sys
import os
import time

# Ensure OOPAO is in path
sys.path.append('/home/yjh/OOPAO')

import numpy as np
import matplotlib.pyplot as plt
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Zernike import Zernike


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def zernike_radial_explicit(n, m, r):
    """
    Explicit calculation of Zernike Radial Polynomial R_n^m(r)
    Formula:
    R_n^m(r) = Sum_{k=0}^{(n-m)/2} [(-1)^k * (n-k)!] / [k! * ((n+m)/2 - k)! * ((n-m)/2 - k)!] * r^(n-2k)
    """
    R = np.zeros_like(r)
    
    # Check parity
    if (n - m) % 2 != 0:
        return R  # R is zero if n-m is odd
        
    for k in range((n - m) // 2 + 1):
        # Coefficients
        num = ((-1)**k) * np.math.factorial(n - k)
        denom = (np.math.factorial(k) * 
                 np.math.factorial((n + m) // 2 - k) * 
                 np.math.factorial((n - m) // 2 - k))
        
        R += (num / denom) * (r**(n - 2 * k))
        
    return R


def zernike_mode_explicit(n, m, X, Y, D):
    """
    Generates a Zernike mode Z_n^m on the grid (X, Y)
    """
    # Normalized coordinates
    R = np.sqrt(X**2 + Y**2) / (D / 2)
    Theta = np.arctan2(Y, X)
    
    # Mask outside pupil
    mask = R <= 1.0
    
    # Initialize Z
    Z = np.zeros_like(X)
    
    # Calculate R_nm only inside pupil
    R_vals = R[mask]
    Theta_vals = Theta[mask]
    
    # Radial function values
    Rad = np.zeros_like(R_vals)
    if (n - m) % 2 == 0:
        for k in range((n - m) // 2 + 1):
            if (n - k) < 0 or ((n + m) // 2 - k) < 0 or ((n - m) // 2 - k) < 0:
                continue
            num = ((-1)**k) * np.math.factorial(n - k)
            denom = (np.math.factorial(k) * 
                     np.math.factorial((n + m) // 2 - k) * 
                     np.math.factorial((n - m) // 2 - k))
            Rad += (num / denom) * (R_vals**(n - 2 * k))
            
    # Azimuthal part
    if m == 0:
        Z[mask] = np.sqrt(n + 1) * Rad
    elif m > 0:
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.cos(m * Theta_vals)
    else:  # m < 0
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.sin(-m * Theta_vals)
        
    return Z


# =============================================================================
# COMPONENT 1: LOAD AND PREPROCESS DATA
# =============================================================================

def load_and_preprocess_data(resolution, diameter, sampling_time, central_obstruction,
                             opt_band, magnitude, n_zernike_modes, n_iterations,
                             r0, L0, wind_speed, wind_direction, altitude):
    """
    Initialize all simulation components: telescope, source, atmosphere, and Zernike basis.
    
    Returns:
        data_dict: Dictionary containing all initialized objects and precomputed data
    """
    print("\n[1] Initializing System...")
    
    # Initialize telescope
    tel = Telescope(resolution=resolution, diameter=diameter, 
                    samplingTime=sampling_time, centralObstruction=central_obstruction)
    
    # Initialize source
    ngs = Source(optBand=opt_band, magnitude=magnitude)
    ngs * tel
    
    print("\n[2] Generating Zernike Basis Explicitly...")
    
    # Create coordinate grid
    y, x = np.indices((tel.resolution, tel.resolution))
    y = (y - tel.resolution / 2) * tel.pixelSize
    x = (x - tel.resolution / 2) * tel.pixelSize
    
    # Generate first few modes explicitly for demonstration
    n_explicit_modes = 6
    zernike_basis_2d = np.zeros((n_explicit_modes, tel.resolution, tel.resolution))
    
    # Mapping Noll Index (j) to (n, m)
    noll_indices = [
        (0, 0),   # Piston (j=1)
        (1, 1),   # Tilt X (j=2)
        (1, -1),  # Tilt Y (j=3)
        (2, 0),   # Defocus (j=4)
        (2, -2),  # Astigmatism (j=5)
        (2, 2),   # Astigmatism (j=6)
    ]
    
    print("    Generating modes using explicit radial polynomials...")
    for j, (n, m) in enumerate(noll_indices):
        mode = zernike_mode_explicit(n, m, x, y, tel.D)
        zernike_basis_2d[j] = mode
        
    # Use OOPAO for the full set to ensure coverage for decomposition
    Z = Zernike(telObject=tel, J=n_zernike_modes)
    Z.computeZernike(tel)
    Z_inv = np.linalg.pinv(Z.modes)  # Pseudoinverse of the basis
    
    # Initialize atmosphere
    print("\n[3] Initializing Atmosphere...")
    atm = Atmosphere(telescope=tel, r0=r0, L0=L0, 
                     fractionalR0=[1], windSpeed=[wind_speed], 
                     windDirection=[wind_direction], altitude=[altitude])
    atm.initializeAtmosphere(tel)
    
    # Create phase map for forward model demonstration
    # 0.5 rad of Defocus (index 3) + 0.5 rad of Astigmatism (index 5)
    phase_map = 0.5 * zernike_basis_2d[3] + 0.5 * zernike_basis_2d[5]
    
    # Convert to OPD [m] for consistency
    # Phase = 2*pi*OPD / lambda => OPD = Phase * lambda / (2*pi)
    opd_map = phase_map * ngs.wavelength / (2 * np.pi)
    
    data_dict = {
        'telescope': tel,
        'source': ngs,
        'atmosphere': atm,
        'zernike_object': Z,
        'zernike_inverse': Z_inv,
        'zernike_basis_2d': zernike_basis_2d,
        'phase_map': phase_map,
        'opd_map': opd_map,
        'coordinate_x': x,
        'coordinate_y': y,
        'n_iterations': n_iterations,
    }
    
    return data_dict


# =============================================================================
# COMPONENT 2: FORWARD OPERATOR
# =============================================================================

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


# =============================================================================
# COMPONENT 3: RUN INVERSION
# =============================================================================

def run_inversion(data_dict):
    """
    Performs iterative Zernike decomposition of atmospheric turbulence.
    
    The inverse model decomposes the OPD map into Zernike coefficients:
    c = Z^+ * phi
    
    Then reconstructs: opd_rec = Z * c
    
    Args:
        data_dict: Dictionary containing telescope, atmosphere, Zernike basis, etc.
        
    Returns:
        results_dict: Dictionary containing inversion results
    """
    print("\n[4] Inverse Model: Decomposing Atmospheric Turbulence...")
    
    tel = data_dict['telescope']
    atm = data_dict['atmosphere']
    Z = data_dict['zernike_object']
    Z_inv = data_dict['zernike_inverse']
    n_iter = data_dict['n_iterations']
    
    rmse_history = []
    all_coeffs = []
    all_opd_original = []
    all_opd_reconstructed = []
    
    for i in range(n_iter):
        # Update atmosphere
        atm.update()
        current_opd = atm.OPD.copy()
        
        # Explicit Decomposition Step
        # Extract valid phase points inside pupil
        opd_masked = current_opd[np.where(tel.pupil == 1)]
        
        # Project onto Zernike Basis (Least Squares fitting)
        # c = Z_dagger * opd
        coeffs = Z_inv @ opd_masked
        
        # Reconstruction
        # opd_rec = Z * c
        reconstructed_opd = np.squeeze(Z.modesFullRes @ coeffs)
        
        # Error calculation
        diff = (current_opd - reconstructed_opd) * tel.pupil
        rmse = np.std(diff[tel.pupil == 1])
        rmse_history.append(rmse)
        
        # Store results
        all_coeffs.append(coeffs.copy())
        all_opd_original.append(current_opd.copy())
        all_opd_reconstructed.append(reconstructed_opd.copy())
        
        print(f"    Iter {i+1}: Fitting RMSE = {rmse * 1e9:.1f} nm")
    
    results_dict = {
        'rmse_history': np.array(rmse_history),
        'all_coeffs': all_coeffs,
        'all_opd_original': all_opd_original,
        'all_opd_reconstructed': all_opd_reconstructed,
        'final_rmse': rmse_history[-1],
        'mean_rmse': np.mean(rmse_history),
    }
    
    return results_dict


# =============================================================================
# COMPONENT 4: EVALUATE RESULTS
# =============================================================================

def evaluate_results(data_dict, inversion_results, output_dir='.'):
    """
    Evaluates and visualizes the results of forward modeling and inversion.
    
    Args:
        data_dict: Dictionary containing input data and forward model results
        inversion_results: Dictionary containing inversion results
        output_dir: Directory to save output figures
    """
    print("\n[5] Evaluating Results...")
    
    tel = data_dict['telescope']
    phase_map = data_dict['phase_map']
    rmse_history = inversion_results['rmse_history']
    
    # Compute PSF using forward operator
    print("    Computing PSF via FFT...")
    psf = forward_operator(phase_map, tel)
    
    # Plot forward model results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(phase_map * tel.pupil)
    plt.title("Input Phase (Explicit Zernikes)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(psf + 1e-10))
    plt.title("Resulting PSF (Log)")
    plt.colorbar()
    forward_plot_path = os.path.join(output_dir, "zernike_forward.png")
    plt.savefig(forward_plot_path)
    plt.close()
    print(f"    Saved {forward_plot_path}")
    
    # Plot inversion results
    plt.figure()
    plt.plot(rmse_history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE [m]')
    plt.title('Zernike Decomposition Residual')
    inverse_plot_path = os.path.join(output_dir, "zernike_inverse.png")
    plt.savefig(inverse_plot_path)
    plt.close()
    print(f"    Saved {inverse_plot_path}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"    Mean RMSE: {inversion_results['mean_rmse'] * 1e9:.2f} nm")
    print(f"    Final RMSE: {inversion_results['final_rmse'] * 1e9:.2f} nm")
    print(f"    Min RMSE: {np.min(rmse_history) * 1e9:.2f} nm")
    print(f"    Max RMSE: {np.max(rmse_history) * 1e9:.2f} nm")
    print(f"    Std RMSE: {np.std(rmse_history) * 1e9:.2f} nm")
    
    # Compare original vs reconstructed OPD for last iteration
    if len(inversion_results['all_opd_original']) > 0:
        last_original = inversion_results['all_opd_original'][-1]
        last_reconstructed = inversion_results['all_opd_reconstructed'][-1]
        
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(last_original * tel.pupil)
        plt.title("Original OPD")
        plt.colorbar(label='[m]')
        
        plt.subplot(1, 3, 2)
        plt.imshow(last_reconstructed * tel.pupil)
        plt.title("Reconstructed OPD")
        plt.colorbar(label='[m]')
        
        plt.subplot(1, 3, 3)
        residual = (last_original - last_reconstructed) * tel.pupil
        plt.imshow(residual)
        plt.title("Residual")
        plt.colorbar(label='[m]')
        
        comparison_path = os.path.join(output_dir, "opd_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"    Saved {comparison_path}")
    
    return {
        'psf': psf,
        'forward_plot_path': forward_plot_path,
        'inverse_plot_path': inverse_plot_path,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=================================================================")
    print("   Explicit Image Formation & Zernike Decomposition (Refactored)")
    print("=================================================================")
    
    # Define parameters
    resolution = 120
    diameter = 8
    sampling_time = 1 / 1000
    central_obstruction = 0.0
    opt_band = 'I'
    magnitude = 10
    n_zernike_modes = 100
    n_iterations = 10
    r0 = 0.15
    L0 = 25
    wind_speed = 10
    wind_direction = 0
    altitude = 0
    
    # Step 1: Load and preprocess data
    data_dict = load_and_preprocess_data(
        resolution=resolution,
        diameter=diameter,
        sampling_time=sampling_time,
        central_obstruction=central_obstruction,
        opt_band=opt_band,
        magnitude=magnitude,
        n_zernike_modes=n_zernike_modes,
        n_iterations=n_iterations,
        r0=r0,
        L0=L0,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        altitude=altitude
    )
    
    # Step 2: Forward operator demonstration (compute PSF)
    print("\n[3] Forward Model: Generating PSF from specific aberrations...")
    psf = forward_operator(data_dict['phase_map'], data_dict['telescope'])
    print(f"    PSF computed with shape: {psf.shape}")
    print(f"    PSF max value: {psf.max():.6f}")
    
    # Step 3: Run inversion
    inversion_results = run_inversion(data_dict)
    
    # Step 4: Evaluate results
    evaluation_results = evaluate_results(data_dict, inversion_results)
    
    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")