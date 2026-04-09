import json

import os

import warnings

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

warnings.filterwarnings('ignore')

def forward_operator(x, ray_transform):
    """
    Apply the forward operator (ray transform / projection) to an image.
    
    This implements the CT forward model: y = Ax, where A is the ray transform
    that computes line integrals through the image along each projection angle.
    
    Parameters
    ----------
    x : numpy.ndarray or odl.DiscretizedSpaceElement
        Input image to project.
    ray_transform : odl.tomo.RayTransform
        The ray transform operator.
    
    Returns
    -------
    numpy.ndarray
        The computed sinogram (projection data).
    """
    # If input is a numpy array, convert to ODL element
    if isinstance(x, np.ndarray):
        x_element = ray_transform.domain.element(x)
    else:
        x_element = x
    
    # Apply ray transform
    y_pred = ray_transform(x_element)
    
    return y_pred.asarray()

def evaluate_results(ground_truth, reconstructions, data_params, output_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR and SSIM metrics for all reconstruction methods,
    saves numerical results and visualizations.
    
    Parameters
    ----------
    ground_truth : numpy.ndarray
        Ground truth image.
    reconstructions : dict
        Dictionary of reconstructions from run_inversion containing:
        - 'fbp': FBP reconstruction
        - 'cgls': CGLS reconstruction
        - 'pdhg': TV-PDHG reconstruction
        - 'parameters': algorithm parameters
    data_params : dict
        Data parameters from load_and_preprocess_data.
    output_dir : str
        Directory to save output files.
    
    Returns
    -------
    dict
        Dictionary of metrics for all methods.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def compute_metrics(gt_arr, recon_arr, label):
        """Compute PSNR and SSIM between ground truth and reconstruction."""
        data_range = gt_arr.max() - gt_arr.min()
        psnr = float(peak_signal_noise_ratio(gt_arr, recon_arr, data_range=data_range))
        ssim = float(structural_similarity(gt_arr, recon_arr, data_range=data_range))
        print(f'  {label:12s}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}')
        return psnr, ssim
    
    print('\n=== Evaluation ===')
    
    recon_fbp = reconstructions['fbp']
    recon_cgls = reconstructions['cgls']
    recon_pdhg = reconstructions['pdhg']
    
    psnr_fbp, ssim_fbp = compute_metrics(ground_truth, recon_fbp, 'FBP')
    psnr_cgls, ssim_cgls = compute_metrics(ground_truth, recon_cgls, 'CGLS')
    psnr_pdhg, ssim_pdhg = compute_metrics(ground_truth, recon_pdhg, 'TV-PDHG')
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_pdhg)
    
    # Compile metrics
    algo_params = reconstructions['parameters']
    metrics = {
        'task': 'odl_inverse',
        'method': 'TV-PDHG (Total Variation via Primal-Dual Hybrid Gradient)',
        'PSNR': round(psnr_pdhg, 4),
        'SSIM': round(ssim_pdhg, 4),
        'all_methods': {
            'FBP': {'PSNR': round(psnr_fbp, 4), 'SSIM': round(ssim_fbp, 4)},
            'CGLS': {'PSNR': round(psnr_cgls, 4), 'SSIM': round(ssim_cgls, 4)},
            'TV-PDHG': {'PSNR': round(psnr_pdhg, 4), 'SSIM': round(ssim_pdhg, 4)},
        },
        'parameters': {
            'image_size': data_params['image_size'],
            'num_angles': data_params['num_angles'],
            'noise_level': data_params['noise_level'],
            'tv_lambda': algo_params['tv_lambda'],
            'pdhg_iterations': algo_params['niter_pdhg'],
        }
    }
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(f'\nMetrics saved → {os.path.join(output_dir, "metrics.json")}')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    gt = ground_truth
    num_angles = data_params['num_angles']
    noise_level = data_params['noise_level']
    
    # Row 1: images
    ax = axes[0, 0]
    im = ax.imshow(gt, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title('Ground Truth (Shepp-Logan)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[0, 1]
    # Create a placeholder for sinogram visualization using forward operator
    sinogram_display = forward_operator(ground_truth, data_params['ray_transform'])
    im = ax.imshow(sinogram_display, cmap='gray', aspect='auto')
    ax.set_title(f'Sinogram ({num_angles} angles, {noise_level*100:.0f}% noise)', fontsize=12)
    ax.set_xlabel('Detector pixel')
    ax.set_ylabel('Angle index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[0, 2]
    im = ax.imshow(recon_fbp, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title(f'FBP  (PSNR={psnr_fbp:.1f} dB, SSIM={ssim_fbp:.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 0]
    im = ax.imshow(recon_cgls, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title(f'CGLS  (PSNR={psnr_cgls:.1f} dB, SSIM={ssim_cgls:.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 1]
    im = ax.imshow(recon_pdhg, cmap='gray', vmin=gt.min(), vmax=gt.max())
    ax.set_title(f'TV-PDHG  (PSNR={psnr_pdhg:.1f} dB, SSIM={ssim_pdhg:.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Error map for TV-PDHG
    ax = axes[1, 2]
    err = np.abs(gt - recon_pdhg)
    im = ax.imshow(err, cmap='hot')
    ax.set_title('|GT − TV-PDHG| Error Map', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle('Task 188: ODL Inverse — CT Reconstruction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure saved  → {os.path.join(output_dir, "reconstruction_result.png")}')
    
    print('\n✓ Task 188 (odl_inverse) completed successfully.')
    print(f'  Primary result (TV-PDHG): PSNR={psnr_pdhg:.2f} dB, SSIM={ssim_pdhg:.4f}')
    
    return metrics
