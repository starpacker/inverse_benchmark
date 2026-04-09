import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(recon_vec, mesh_truth, mesh_baseline, grid_info, out_name='reconstruction_result.png'):
    """
    Computes metrics and plots results.
    """
    grid_shape = grid_info['shape']
    xgrid = grid_info['xgrid']
    ygrid = grid_info['ygrid']
    
    # Reconstruct Image
    recon_img = recon_vec.reshape(grid_shape)
    
    # Ground Truth Image
    dmua_mesh = mesh_truth.mua - mesh_baseline.mua
    dmua_truth_grid = grid_info['mesh2grid'] @ dmua_mesh
    truth_img = dmua_truth_grid.reshape(grid_shape)
    
    # Metrics
    mse = np.mean((recon_vec - dmua_truth_grid)**2)
    max_val = np.max(np.abs(dmua_truth_grid))
    if max_val == 0: max_val = 1.0
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-16))
    
    print(f"Evaluation Metrics:")
    print(f"  MSE: {mse:.6e}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    
    vmax = np.max(np.abs(truth_img))
    if vmax == 0: vmax = 1.0
    
    im1 = ax1.imshow(truth_img, origin='lower', extent=extent, cmap='jet', vmin=-vmax, vmax=vmax)
    ax1.set_title('Ground Truth ($\Delta \mu_a$)')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(recon_img, origin='lower', extent=extent, cmap='jet', vmin=-vmax, vmax=vmax)
    ax2.set_title(f'Reconstruction (PSNR: {psnr:.2f} dB)')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle("NIRFASTer Reconstruction Demo")
    plt.tight_layout()
    plt.savefig(out_name)
    print(f"Plot saved to {out_name}")
    
    return mse, psnr
