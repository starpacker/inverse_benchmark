import os
import sys
import torch

# Ensure the DPItorch submodule is accessible if located in a subdirectory
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

# Set default precision to float32
torch.set_default_dtype(torch.float32)

def fft2c_torch(img):
    """
    2D FFT for torch tensors, returns real/imag stacked on last dim.
    
    Args:
        img: Input tensor (usually real-valued).
        
    Returns:
        Tensor with shape (..., 2) representing (Real, Imaginary) parts of the FFT.
    """
    # 1. Add a dimension for the imaginary part
    x = img.unsqueeze(-1)
    
    # 2. Create a zero-filled imaginary component and stack it
    x = torch.cat([x, torch.zeros_like(x)], -1)
    
    # 3. Convert (..., 2) float tensor to native complex tensor
    xc = torch.view_as_complex(x)
    
    # 4. Perform 2D FFT with orthogonal normalization
    kc = torch.fft.fft2(xc, norm="ortho")
    
    # 5. Convert back to (..., 2) float tensor
    return torch.view_as_real(kc)

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
    # Compute full k-space via FFT
    kspace_pred = fft2c_torch(img)
    
    # Apply undersampling mask
    kspace_masked = kspace_pred * mask_tensor
    
    return kspace_masked

if __name__ == "__main__":
    # 1. Create a dummy image (Batch=1, Size=32x32)
    N = 32
    dummy_img = torch.randn(1, N, N)
    
    # 2. Create a full mask (all ones) to test pure FFT behavior
    # Shape must be (N, N, 2) to broadcast over batch and match complex dim
    full_mask = torch.ones(N, N, 2)
    
    # 3. Run forward operator
    kspace_out = forward_operator(dummy_img, full_mask)
    
    # 4. Check shapes
    expected_shape = (1, N, N, 2)
    assert kspace_out.shape == expected_shape, \
        f"Shape mismatch: Expected {expected_shape}, got {kspace_out.shape}"
    
    # 5. Check Parseval's Theorem (Energy conservation with norm='ortho')
    # Energy in image domain
    energy_img = torch.sum(dummy_img ** 2)
    # Energy in k-space (sum of squares of real and imag parts)
    energy_kspace = torch.sum(kspace_out ** 2)
    
    # Allow small numerical tolerance
    assert torch.allclose(energy_img, energy_kspace, atol=1e-5), \
        "Energy not conserved! Check FFT normalization."
        
    print("Forward Operator implementation verified successfully.")