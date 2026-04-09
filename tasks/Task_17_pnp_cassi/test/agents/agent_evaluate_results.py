import os
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def psnr(ref, img):
    """
    Peak signal-to-noise ratio (PSNR).
    
    Args:
        ref (np.ndarray): The ground truth image.
        img (np.ndarray): The reconstructed image.
        
    Returns:
        float: The PSNR value in decibels (dB).
    """
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def evaluate_results(recon_img, truth, psnrs, output_dir='.'):
    """
    Evaluate and save reconstruction results.
    
    Args:
        recon_img (np.ndarray): The final reconstructed 3D spectral cube.
        truth (np.ndarray): The ground truth 3D spectral cube.
        psnrs (list): A list of PSNR values recorded per iteration.
        output_dir (str): Directory to save outputs. Defaults to current dir.
        
    Returns:
        float: The final PSNR value.
    """
    # Final PSNR
    final_psnr = psnr(truth, recon_img)
    print(f"Final PSNR: {final_psnr:.2f} dB")

    # Save Reconstruction as .mat
    sio.savemat(os.path.join(output_dir, 'recon_result.mat'), {'img': recon_img})

    # Save spectral channels as image grid
    nC = recon_img.shape[2]
    fig = plt.figure(figsize=(10, 10))
    
    for i in range(min(9, nC)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(recon_img[:, :, i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'Band {i+1}')
        
    plt.savefig(os.path.join(output_dir, 'recon_channels.png'))
    plt.close()

    # Save PSNR plot
    plt.figure()
    plt.plot(psnrs)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('Reconstruction Convergence')
    plt.savefig(os.path.join(output_dir, 'psnr_curve.png'))
    plt.close()

    return final_psnr