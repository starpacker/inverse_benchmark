# dpi_task2.py
```python
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import math
import cv2
import argparse

# Add DPItorch to path for imports
# this is very important
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

from generative_model import realnvpfc_model

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Inlined MRI_helpers.py
# ==============================================================================

def Loss_kspace_diff(sigma):
    def func(y_true, y_pred):
        return torch.mean(torch.abs(y_pred - y_true), (1, 2, 3)) / sigma
    return func

def Loss_kspace_diff2(sigma):
    def func(y_true, y_pred):
        return torch.mean((y_pred - y_true)**2, (1, 2, 3)) / (sigma)**2
    return func

def Loss_l1(y_pred):
    return torch.mean(torch.abs(y_pred), (-1, -2))

def Loss_TSV(y_pred):
    return torch.mean((y_pred[:, 1::, :] - y_pred[:, 0:-1, :])**2, (-1, -2)) + torch.mean((y_pred[:, :, 1::] - y_pred[:, :, 0:-1])**2, (-1, -2))

def Loss_TV(y_pred):
    return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))

# ==============================================================================
# Main Execution (DPI_MRI.py Logic)
# ==============================================================================

def readMRIdata(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
        target_image = obj['target']
    return target_image

def fft2c(data):
    data = np.fft.fft2(data, norm="ortho")
    return np.stack((data.real, data.imag), axis=-1)

def fft2c_torch(img):
    x = img.unsqueeze(-1)
    x = torch.cat([x, torch.zeros_like(x)], -1)
    # Use modern torch.fft if available, else compat
    if hasattr(torch.fft, 'fft2'):
        # Complex tensor in new pytorch is 1 channel complex
        # But here x is (B, H, W, 2)
        # We need to treat last dim as complex
        xc = torch.view_as_complex(x)
        kc = torch.fft.fft2(xc, norm="ortho")
        return torch.view_as_real(kc)
    else:
        # Fallback for old torch (1.3 style used in original code)
        # But we are using modern torch, so new API is better.
        # Original code used torch.fft(x, 2, normalized=True) which is deprecated/removed
        # We implement modern replacement
        xc = torch.view_as_complex(x)
        kc = torch.fft.fft2(xc, norm="ortho")
        return torch.view_as_real(kc)

class Img_logscale(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        log_scale = torch.Tensor(np.log(scale)*np.ones(1))
        self.log_scale = nn.Parameter(log_scale)
    def forward(self):
        return self.log_scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPI Task 2: MRI")
    parser.add_argument("--impath", default='dataset/fastmri_sample/mri/knee/scan_0.pkl', type=str)
    parser.add_argument("--maskpath", default='dataset/fastmri_sample/mask/mask4.npy', type=str)
    parser.add_argument("--save_path", default='./checkpoints', type=str)
    parser.add_argument("--npix", default=64, type=int)
    parser.add_argument("--n_flow", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--n_epoch", default=10, type=int) # Reduced to 10
    parser.add_argument("--logdet", default=1.0, type=float)
    parser.add_argument("--sigma", default=5e-7, type=float)
    parser.add_argument("--l1", default=0.0, type=float)
    parser.add_argument("--tv", default=1e3, type=float)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load Data
    img_true = readMRIdata(args.impath)
    img_true = cv2.resize(img_true, (args.npix, args.npix), interpolation=cv2.INTER_AREA)
    kspace = fft2c(img_true)
    kspace = kspace + np.random.normal(size=kspace.shape) * args.sigma
    mask = np.load(args.maskpath)
    
    # Adjust mask size if needed (Original code hardcodes [24:40, 24:40] for center, let's keep logic but ensure fit)
    # Mask usually matches image size? Original code: mask = np.load(args.maskpath) which is 320x320?
    # We need to resize mask to npix?
    # Original code doesn't resize mask explicitly, assumes mask file matches?
    # Wait, mask4.npy might be 320x320. If npix=64, we need to crop or resize mask.
    # Let's check mask shape. If large, crop center.
    if mask.shape[0] != args.npix:
        # Simple resize or crop. Let's resize.
        mask = cv2.resize(mask, (args.npix, args.npix), interpolation=cv2.INTER_NEAREST)
        
    mask[int(args.npix/2)-8:int(args.npix/2)+8, int(args.npix/2)-8:int(args.npix/2)+8] = 1
    mask = np.fft.fftshift(mask)
    mask = np.stack((mask, mask), axis=-1)

    args.flux = np.sum(img_true)

    # Model
    img_generator = realnvpfc_model.RealNVP(args.npix*args.npix, args.n_flow, affine=True).to(device)
    logscale_factor = Img_logscale(scale=args.flux/(0.8*args.npix*args.npix)).to(device)

    Loss_kspace_img = Loss_kspace_diff2(args.sigma)
    imgl1_weight = args.l1 / args.flux
    imgtv_weight = args.tv * args.npix / args.flux
    logdet_weight = args.logdet / (0.5 * np.sum(mask))

    kspace_true = torch.Tensor(mask * kspace).to(device=device)
    
    optimizer = optim.Adam(list(img_generator.parameters())+list(logscale_factor.parameters()), lr = args.lr)

    print(f"Starting MRI reconstruction for {args.n_epoch} epochs...")
    
    for k in range(args.n_epoch):
        z_sample = torch.randn(64, args.npix*args.npix).to(device=device)
        img_samp, logdet = img_generator.reverse(z_sample)
        img_samp = img_samp.reshape((-1, args.npix, args.npix))
        
        logscale_factor_value = logscale_factor.forward()
        scale_factor = torch.exp(logscale_factor_value)
        img = torch.nn.Softplus()(img_samp) * scale_factor
        
        det_softplus = torch.sum(img_samp - torch.nn.Softplus()(img_samp), (1, 2))
        det_scale = logscale_factor_value * args.npix * args.npix
        logdet = logdet + det_softplus + det_scale
        
        kspace_pred = fft2c_torch(img)
        loss_data = Loss_kspace_img(kspace_true, kspace_pred * torch.Tensor(mask).to(device)) / np.mean(mask)
        
        loss_l1 = Loss_l1(img) if imgl1_weight>0 else 0
        loss_tv = Loss_TV(img) if imgtv_weight>0 else 0
        
        loss_prior = imgtv_weight * loss_tv + imgl1_weight * loss_l1
        
        loss = torch.mean(loss_data) + torch.mean(loss_prior) - logdet_weight*torch.mean(logdet)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(img_generator.parameters())+ list(logscale_factor.parameters()), 1e-2)
        optimizer.step()
        
        if k % 1 == 0:
            print(f"Epoch {k}: Loss {loss.item():.4f}, KSpace {torch.mean(loss_data).item():.4f}")

    print("Saving MRI results...")
    torch.save(img_generator.state_dict(), os.path.join(args.save_path, 'mri_model.pth'))
    np.save(os.path.join(args.save_path, 'mri_reconstruction.npy'), img.detach().cpu().numpy())
    print("DPI Task 2 Finished Successfully")

```