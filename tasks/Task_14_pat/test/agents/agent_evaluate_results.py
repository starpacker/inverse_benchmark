import numpy as np
import matplotlib.pyplot as plt

def evaluate_results(reconstruction, so2, output_file="pat_result.png"):
    """
    Evaluate and visualize the reconstruction and sO2 results.
    
    Args:
        reconstruction: Reconstructed images, shape (n_wl, nz, ny, nx)
        so2: Oxygen saturation map, shape (nz, ny, nx)
        output_file: Path to save the output figure
        
    Returns:
        mean_so2: Mean sO2 value in the ROI
    """
    # 1. Data Preparation: Mean projection across wavelengths and slice extraction
    # reconstruction shape is (n_wl, nz, ny, nx). Mean over axis 0 gives (nz, ny, nx).
    # Taking [0] gives the first z-slice (ny, nx).
    recon_img = np.mean(reconstruction, axis=0)[0]
    so2_img = so2[0]
    
    # 2. Visualization Setup
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Structural Reconstruction
    plt.subplot(1, 2, 1)
    # Transpose and origin='lower' are critical for correct spatial orientation
    plt.imshow(recon_img.T, cmap='gray', origin='lower')
    plt.title("Reconstruction (Mean WL)")
    plt.colorbar(label="PA Signal")
    
    # Subplot 2: Functional sO2 Map
    plt.subplot(1, 2, 2)
    plt.imshow(so2_img.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.title("sO2 Estimation")
    plt.colorbar(label="sO2")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
    
    # 3. Quantitative Metrics
    # Filter for valid sO2 pixels (background is usually 0 or NaN, here assumed > 0)
    valid_so2 = so2_img[so2_img > 0]
    
    if len(valid_so2) > 0:
        mean_so2 = np.mean(valid_so2)
    else:
        mean_so2 = 0.0
    
    print(f"Mean sO2 in ROI: {mean_so2:.4f}")
    
    # Check dynamic range of the reconstruction for stability verification
    recon_max = np.max(reconstruction)
    recon_min = np.min(reconstruction)
    print(f"Reconstruction range: [{recon_min:.4f}, {recon_max:.4f}]")
    
    return mean_so2