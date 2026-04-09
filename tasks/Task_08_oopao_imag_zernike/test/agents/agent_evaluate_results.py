import numpy as np
import matplotlib.pyplot as plt
import os  # CRITICAL: Do not forget this import!

def forward_operator(phase_map, tel):
    # Amplitude
    amplitude = tel.pupil
    
    # Electric Field
    electric_field = amplitude * np.exp(1j * phase_map)
    
    # Padding
    zero_padding = 4
    N = tel.resolution
    N_padded = N * zero_padding
    pad_width = (N_padded - N) // 2
    electric_field_padded = np.pad(electric_field, pad_width, mode='constant')
    
    # FFT
    complex_focal_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(electric_field_padded)))
    
    # PSF
    psf = np.abs(complex_focal_plane)**2
    psf = psf / psf.max()
    
    return psf

def evaluate_results(data_dict, inversion_results, output_dir='.'):
    # Unpacking
    tel = data_dict['telescope']
    phase_map = data_dict['phase_map']
    rmse_history = inversion_results['rmse_history']
    
    # Compute PSF
    psf = forward_operator(phase_map, tel)
    
    # Plot 1 (Forward)
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
    print(f"Saved {forward_plot_path}")
    
    # Plot 2 (Inverse)
    plt.figure()
    plt.plot(rmse_history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE [m]')
    plt.title('Zernike Decomposition Residual')
    
    inverse_plot_path = os.path.join(output_dir, "zernike_inverse.png")
    plt.savefig(inverse_plot_path)
    plt.close()
    print(f"Saved {inverse_plot_path}")
    
    # Statistics
    rmse_nm = np.array(rmse_history) * 1e9
    print(f"Mean RMSE: {rmse_nm.mean():.2f} nm")
    print(f"Final RMSE: {rmse_nm[-1]:.2f} nm")
    print(f"Min RMSE: {rmse_nm.min():.2f} nm")
    print(f"Max RMSE: {rmse_nm.max():.2f} nm")
    print(f"Std RMSE: {rmse_nm.std():.2f} nm")
    
    # Plot 3 (OPD Comparison)
    if len(inversion_results['all_opd_original']) > 0:
        original_opd = inversion_results['all_opd_original'][-1]
        reconstructed_opd = inversion_results['all_opd_reconstructed'][-1]
        
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_opd * tel.pupil)
        plt.title("Original OPD")
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed_opd * tel.pupil)
        plt.title("Reconstructed OPD")
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow((original_opd - reconstructed_opd) * tel.pupil)
        plt.title("Residual")
        plt.colorbar()
        
        comparison_path = os.path.join(output_dir, "opd_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"Saved {comparison_path}")
    
    # Return Dictionary
    return {
        'psf': psf,
        'forward_plot_path': forward_plot_path,
        'inverse_plot_path': inverse_plot_path,
    }