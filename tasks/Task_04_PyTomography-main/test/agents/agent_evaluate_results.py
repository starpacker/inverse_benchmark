import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

def evaluate_results(gt_object, projections_noisy, recon, shape):
    """
    Computes metrics (PSNR, SSIM) and saves comparison plot.
    """
    print("\n=== Evaluation ===")
    gt_np = gt_object.cpu().numpy()
    recon_np = recon.cpu().numpy()
    
    # Normalize for metric calculation
    gt_norm = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min())
    recon_norm = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min())
    
    # Slice for metrics (middle slice)
    mid_z = shape[2] // 2
    gt_slice = gt_norm[:, :, mid_z]
    recon_slice = recon_norm[:, :, mid_z]
    
    p = psnr(gt_slice, recon_slice, data_range=1.0)
    s = ssim(gt_slice, recon_slice, data_range=1.0)
    
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.4f}")
    
    # Save Results
    print("\nSaving results to 'reconstruction_results_spect.png'...")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gt_slice, cmap='gray')
    ax[0].set_title("Ground Truth (Z-slice)")
    ax[1].imshow(projections_noisy.cpu().numpy()[0, :, :].T, cmap='gray') 
    ax[1].set_title("Projection (Angle 0)")
    ax[2].imshow(recon_slice, cmap='gray')
    ax[2].set_title(f"OSEM Recon\nPSNR: {p:.2f}")
    
    plt.tight_layout()
    plt.savefig("reconstruction_results_spect.png")
    print("Done.")
