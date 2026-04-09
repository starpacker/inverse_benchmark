import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import math
import argparse
from PIL import Image

# Add DPItorch to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

from generative_model import realnvpfc_model

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Helper Functions
# ==============================================================================

def resize_array(arr, new_size):
    """Resize 2D array using PIL instead of cv2"""
    img = Image.fromarray(arr.astype(np.float32), mode='F')
    img_resized = img.resize((new_size, new_size), Image.BILINEAR)
    return np.array(img_resized)


def resize_mask(arr, new_size):
    """Resize mask array using nearest neighbor"""
    img = Image.fromarray(arr.astype(np.uint8))
    img_resized = img.resize((new_size, new_size), Image.NEAREST)
    return np.array(img_resized)


def fft2c(data):
    """2D FFT with ortho normalization, returns real/imag stacked"""
    data = np.fft.fft2(data, norm="ortho")
    return np.stack((data.real, data.imag), axis=-1)


def fft2c_torch(img):
    """2D FFT for torch tensors, returns real/imag stacked on last dim"""
    x = img.unsqueeze(-1)
    x = torch.cat([x, torch.zeros_like(x)], -1)
    xc = torch.view_as_complex(x)
    kc = torch.fft.fft2(xc, norm="ortho")
    return torch.view_as_real(kc)


def Loss_kspace_diff2(sigma):
    """K-space L2 loss function"""
    def func(y_true, y_pred):
        return torch.mean((y_pred - y_true)**2, (1, 2, 3)) / (sigma)**2
    return func


def Loss_l1(y_pred):
    """L1 loss on image"""
    return torch.mean(torch.abs(y_pred), (-1, -2))


def Loss_TV(y_pred):
    """Total variation loss"""
    return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + \
           torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))


class Img_logscale(nn.Module):
    """Learnable log scale parameter"""
    def __init__(self, scale=1):
        super().__init__()
        log_scale = torch.Tensor(np.log(scale) * np.ones(1))
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale


# ==============================================================================
# Main Functional Components
# ==============================================================================

def load_and_preprocess_data(impath, maskpath, npix, sigma):
    """
    Load MRI data and mask, preprocess for reconstruction.
    
    Args:
        impath: Path to pickle file containing target MRI image
        maskpath: Path to numpy file containing k-space mask
        npix: Target image size (npix x npix)
        sigma: Noise level for k-space
        
    Returns:
        Dictionary containing:
            - img_true: Ground truth image
            - kspace: Noisy k-space measurements
            - mask: Undersampling mask (with center region set to 1)
            - flux: Total intensity of ground truth
    """
    # Load target image from pickle
    with open(impath, 'rb') as f:
        obj = pickle.load(f)
        img_true = obj['target']
    
    # Resize image to target size
    img_true = resize_array(img_true, npix)
    
    # Compute k-space and add noise
    kspace = fft2c(img_true)
    kspace = kspace + np.random.normal(size=kspace.shape) * sigma
    
    # Load and process mask
    mask = np.load(maskpath)
    if mask.shape[0] != npix:
        mask = resize_mask(mask, npix)
    
    # Ensure center region is fully sampled
    center_size = 8
    center_start = npix // 2 - center_size
    center_end = npix // 2 + center_size
    mask[center_start:center_end, center_start:center_end] = 1
    
    # FFT shift mask and stack for real/imag components
    mask = np.fft.fftshift(mask)
    mask = np.stack((mask, mask), axis=-1)
    
    # Compute flux (total intensity)
    flux = np.sum(img_true)
    
    return {
        'img_true': img_true,
        'kspace': kspace,
        'mask': mask,
        'flux': flux,
        'npix': npix,
        'sigma': sigma
    }


def forward_operator(img, mask_tensor):
    """
    Forward operator: Image -> Masked K-space
    
    Applies 2D FFT to image and multiplies by undersampling mask.
    
    Args:
        img: Image tensor of shape (batch, npix, npix)
        mask_tensor: K-space mask tensor of shape (npix, npix, 2)
        
    Returns:
        Masked k-space tensor of shape (batch, npix, npix, 2)
    """
    # Compute k-space via FFT
    kspace_pred = fft2c_torch(img)
    
    # Apply mask
    kspace_masked = kspace_pred * mask_tensor
    
    return kspace_masked


def run_inversion(data, n_flow, n_epoch, lr, logdet_weight, l1_weight, tv_weight):
    """
    Run MRI reconstruction using Deep Probabilistic Imaging.
    
    Args:
        data: Dictionary from load_and_preprocess_data
        n_flow: Number of flow layers in RealNVP
        n_epoch: Number of training epochs
        lr: Learning rate
        logdet_weight: Weight for log determinant term
        l1_weight: Weight for L1 regularization
        tv_weight: Weight for TV regularization
        
    Returns:
        Dictionary containing:
            - model: Trained generator model
            - logscale: Trained log scale factor
            - reconstructed: Final reconstructed images (numpy array)
            - loss_history: List of loss values
    """
    npix = data['npix']
    sigma = data['sigma']
    flux = data['flux']
    mask = data['mask']
    kspace = data['kspace']
    
    # Initialize model
    img_generator = realnvpfc_model.RealNVP(npix * npix, n_flow, affine=True).to(device)
    logscale_factor = Img_logscale(scale=flux / (0.8 * npix * npix)).to(device)
    
    # Loss function
    Loss_kspace_img = Loss_kspace_diff2(sigma)
    
    # Compute normalized weights
    imgl1_weight = l1_weight / flux
    imgtv_weight = tv_weight * npix / flux
    logdet_w = logdet_weight / (0.5 * np.sum(mask))
    
    # Move data to device
    mask_tensor = torch.Tensor(mask).to(device=device)
    kspace_true = torch.Tensor(mask * kspace).to(device=device)
    mask_mean = np.mean(mask)
    
    # Optimizer
    optimizer = optim.Adam(
        list(img_generator.parameters()) + list(logscale_factor.parameters()),
        lr=lr
    )
    
    loss_history = []
    
    print(f"Starting MRI reconstruction for {n_epoch} epochs...")
    
    for k in range(n_epoch):
        # Sample latent codes
        z_sample = torch.randn(64, npix * npix).to(device=device)
        
        # Generate images via flow
        img_samp, logdet = img_generator.reverse(z_sample)
        img_samp = img_samp.reshape((-1, npix, npix))
        
        # Apply scale factor and softplus activation
        logscale_factor_value = logscale_factor.forward()
        scale_factor = torch.exp(logscale_factor_value)
        img = torch.nn.Softplus()(img_samp) * scale_factor
        
        # Compute Jacobian correction for softplus and scale
        det_softplus = torch.sum(img_samp - torch.nn.Softplus()(img_samp), (1, 2))
        det_scale = logscale_factor_value * npix * npix
        logdet = logdet + det_softplus + det_scale
        
        # Forward operator: compute masked k-space
        kspace_pred = forward_operator(img, mask_tensor)
        
        # Data fidelity loss
        loss_data = Loss_kspace_img(kspace_true, kspace_pred) / mask_mean
        
        # Regularization losses
        loss_l1 = Loss_l1(img) if imgl1_weight > 0 else 0
        loss_tv = Loss_TV(img) if imgtv_weight > 0 else 0
        
        loss_prior = imgtv_weight * loss_tv + imgl1_weight * loss_l1
        
        # Total loss
        loss = torch.mean(loss_data) + torch.mean(loss_prior) - logdet_w * torch.mean(logdet)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(img_generator.parameters()) + list(logscale_factor.parameters()),
            1e-2
        )
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if k % 1 == 0:
            print(f"Epoch {k}: Loss {loss.item():.4f}, KSpace {torch.mean(loss_data).item():.4f}")
    
    # Generate final reconstruction
    with torch.no_grad():
        z_sample = torch.randn(64, npix * npix).to(device=device)
        img_samp, _ = img_generator.reverse(z_sample)
        img_samp = img_samp.reshape((-1, npix, npix))
        logscale_factor_value = logscale_factor.forward()
        scale_factor = torch.exp(logscale_factor_value)
        final_img = torch.nn.Softplus()(img_samp) * scale_factor
        reconstructed = final_img.detach().cpu().numpy()
    
    return {
        'model': img_generator,
        'logscale': logscale_factor,
        'reconstructed': reconstructed,
        'loss_history': loss_history
    }


def evaluate_results(data, results, save_path):
    """
    Evaluate reconstruction results and save outputs.
    
    Args:
        data: Dictionary from load_and_preprocess_data
        results: Dictionary from run_inversion
        save_path: Directory to save results
        
    Returns:
        Dictionary containing:
            - mean_reconstruction: Mean of reconstructed samples
            - rmse: Root mean squared error vs ground truth
            - psnr: Peak signal-to-noise ratio
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_true = data['img_true']
    reconstructed = results['reconstructed']
    
    # Compute mean reconstruction
    mean_reconstruction = np.mean(reconstructed, axis=0)
    
    # Compute RMSE
    mse = np.mean((mean_reconstruction - img_true) ** 2)
    rmse = np.sqrt(mse)
    
    # Compute PSNR
    max_val = np.max(img_true)
    if mse > 0:
        psnr = 20 * np.log10(max_val / rmse)
    else:
        psnr = float('inf')
    
    # Save model and reconstruction
    torch.save(results['model'].state_dict(), os.path.join(save_path, 'mri_model.pth'))
    np.save(os.path.join(save_path, 'mri_reconstruction.npy'), reconstructed)
    np.save(os.path.join(save_path, 'mri_mean_reconstruction.npy'), mean_reconstruction)
    
    # Print metrics
    print(f"Reconstruction RMSE: {rmse:.6f}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    print(f"Saved model to {os.path.join(save_path, 'mri_model.pth')}")
    print(f"Saved reconstruction to {os.path.join(save_path, 'mri_reconstruction.npy')}")
    
    return {
        'mean_reconstruction': mean_reconstruction,
        'rmse': rmse,
        'psnr': psnr
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPI Task 2: MRI")
    parser.add_argument("--impath", default='dataset/fastmri_sample/mri/knee/scan_0.pkl', type=str)
    parser.add_argument("--maskpath", default='dataset/fastmri_sample/mask/mask4.npy', type=str)
    parser.add_argument("--save_path", default='./checkpoints', type=str)
    parser.add_argument("--npix", default=64, type=int)
    parser.add_argument("--n_flow", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--n_epoch", default=10, type=int)
    parser.add_argument("--logdet", default=1.0, type=float)
    parser.add_argument("--sigma", default=5e-7, type=float)
    parser.add_argument("--l1", default=0.0, type=float)
    parser.add_argument("--tv", default=1e3, type=float)
    
    args = parser.parse_args()
    
    # Step 1: Load and preprocess data
    data = load_and_preprocess_data(
        impath=args.impath,
        maskpath=args.maskpath,
        npix=args.npix,
        sigma=args.sigma
    )
    
    # Step 2 & 3: Run inversion (forward_operator is called inside)
    results = run_inversion(
        data=data,
        n_flow=args.n_flow,
        n_epoch=args.n_epoch,
        lr=args.lr,
        logdet_weight=args.logdet,
        l1_weight=args.l1,
        tv_weight=args.tv
    )
    
    # Step 4: Evaluate results
    metrics = evaluate_results(
        data=data,
        results=results,
        save_path=args.save_path
    )
    
    print("DPI Task 2 Finished Successfully")
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")