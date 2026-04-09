import matplotlib.pyplot as plt

def evaluate_results(ground_truth, recon_ph, data_ISM_noise):
    """
    Calculates PSNR/SSIM and saves result plot.
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    print("Evaluating Results...")
    
    # Normalize images for metrics calculation
    gt_norm = ground_truth[0] / ground_truth[0].max()
    recon_norm = recon_ph[0] / recon_ph[0].max()
    ism_norm = data_ISM_noise.sum(-1) / data_ISM_noise.sum(-1).max()

    # Calculate PSNR and SSIM
    psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)
    
    print(f"Reconstruction (In-Focus) vs Ground Truth:")
    print(f"PSNR: {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")

    psnr_ism = psnr(gt_norm, ism_norm, data_range=1.0)
    ssim_ism = ssim(gt_norm, ism_norm, data_range=1.0)
    print(f"Raw ISM Sum (Confocal-like) vs Ground Truth:")
    print(f"PSNR: {psnr_ism:.4f}")
    print(f"SSIM: {ssim_ism:.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ground_truth[0], cmap='magma')
    axes[0].set_title('Ground Truth (In-Focus)')
    axes[1].imshow(data_ISM_noise.sum(-1), cmap='magma')
    axes[1].set_title('Raw ISM Sum')
    axes[2].imshow(recon_ph[0], cmap='magma')
    axes[2].set_title('s2ISM Reconstruction')
    plt.tight_layout()
    plt.savefig('s2ism_result.png')
    print("Result image saved to s2ism_result.png")
