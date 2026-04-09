import torch
import torch.nn as nn
import numpy as np

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
        # Ensure scale is positive and non-zero to avoid log(0)
        scale = max(scale, 1e-6)
        log_scale = torch.tensor(np.log(scale) * np.ones(1), dtype=torch.float32)
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale

def forward_operator(img, mask_tensor):
    """
    Forward operator: Image -> Masked K-space
    """
    # Compute k-space via FFT
    kspace_pred = fft2c_torch(img)
    
    # Apply mask
    kspace_masked = kspace_pred * mask_tensor
    
    return kspace_masked

def run_inversion(data, n_flow, n_epoch, lr, logdet_weight, l1_weight, tv_weight):
    """
    Run MRI reconstruction using Deep Probabilistic Imaging.
    """
    npix = data['npix']
    sigma = data['sigma']
    flux = data['flux']
    mask = data['mask']
    kspace = data['kspace']
    
    # Initialize model (Using the imported mock or real model)
    img_generator = realnvpfc_model.RealNVP(npix * npix, n_flow, affine=True).to(device)
    logscale_factor = Img_logscale(scale=flux / (0.8 * npix * npix)).to(device)
    
    # Loss function
    Loss_kspace_img = Loss_kspace_diff2(sigma)
    
    # Compute normalized weights
    imgl1_weight = l1_weight / flux
    imgtv_weight = tv_weight * npix / flux
    
    # Safety check for mask sum to avoid division by zero
    mask_sum = np.sum(mask)
    logdet_w = logdet_weight / (0.5 * mask_sum) if mask_sum > 0 else 0
    
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
        z_sample = torch.randn(2, npix * npix).to(device=device) # Batch size 2 for efficiency
        
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
        
        if k % 10 == 0:
            print(f"Epoch {k}: Loss {loss.item():.4f}")
    
    # Generate final reconstruction
    with torch.no_grad():
        z_sample = torch.randn(2, npix * npix).to(device=device)
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