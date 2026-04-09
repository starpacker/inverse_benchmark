import torch
import matplotlib.pyplot as plt

def evaluate_results(reconstruction, ground_truth, save_figures=True):
    """
    Computes MSE and PSNR, and optionally saves visualization figures.
    
    Args:
        reconstruction: 3D Tensor [nz, ny, nx] - reconstructed volume
        ground_truth: 3D Tensor [nz, ny, nx] - ground truth volume
        save_figures: bool - whether to save visualization figures
    
    Returns:
        metrics: dict containing MSE and PSNR values
    """
    # Compute MSE
    mse = torch.mean((reconstruction - ground_truth) ** 2)
    
    # Compute PSNR
    max_val = ground_truth.max()
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    mse_val = mse.item()
    psnr_val = psnr.item()
    
    print(f"Result: MSE={mse_val:.6f}, PSNR={psnr_val:.2f} dB")
    
    if save_figures:
        # Max projection along Z
        recon_mip = reconstruction.max(dim=0)[0].cpu().numpy()
        gt_mip = ground_truth.max(dim=0)[0].cpu().numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt_mip, cmap='inferno')
        ax[0].set_title("Ground Truth (MIP)")
        ax[1].imshow(recon_mip, cmap='inferno')
        ax[1].set_title(f"Reconstruction (MIP)\nPSNR={psnr_val:.2f}dB")
        plt.savefig("result_comparison.png")
        plt.close()
        print("Saved result_comparison.png")
    
    return {
        'mse': mse_val,
        'psnr': psnr_val
    }