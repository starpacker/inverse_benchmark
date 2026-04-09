import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(y_true, y_pred, recon_object):
    """
    Calculate metrics (PSNR) and display results.
    """
    # MSE and PSNR calculation
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        pixel_max = max(y_true.max(), y_pred.max())
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

    print(f"Evaluation Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(y_true, cmap='viridis')
    plt.title("Original Projection (Q0)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recon_object, cmap='magma', vmax=recon_object.max()*0.5)
    plt.title("Reconstructed Object")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred, cmap='viridis')
    plt.title(f"Reprojection (Forward)\nPSNR: {psnr:.1f} dB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_results.png')
    print("Results saved to reconstruction_results.png")
    
    return psnr
