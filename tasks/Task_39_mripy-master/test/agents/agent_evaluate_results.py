import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import matplotlib.cm as cm

def plotim3(im, save_path=None):
    im = np.flip(im, 0)
    plt.figure()
    plt.imshow(im, cmap=cm.gray, origin='lower', interpolation='none')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.close()

def evaluate_results(reconstruction, reference_image, save_prefix="result"):
    """
    Computes PSNR and saves images.
    """
    # Squeeze extra dimensions for consistent comparison
    reconstruction = np.squeeze(reconstruction)
    reference_image = np.squeeze(reference_image)
    
    # Take magnitude for complex-valued images
    if np.iscomplexobj(reconstruction):
        reconstruction = np.abs(reconstruction)
    if np.iscomplexobj(reference_image):
        reference_image = np.abs(reference_image)
    
    # Normalize images for fair comparison
    if np.max(reference_image) != 0:
        ref_norm = reference_image / np.max(reference_image)
    else:
        ref_norm = reference_image
        
    if np.max(reconstruction) != 0:
        rec_norm = reconstruction / np.max(reconstruction)
    else:
        rec_norm = reconstruction

    # MSE and PSNR
    mse = np.mean((ref_norm - rec_norm) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    
    # Save images
    plotim3(rec_norm, save_path=f'{save_prefix}_recon.png')
    plotim3(ref_norm, save_path=f'{save_prefix}_ref.png')
    
    return psnr
