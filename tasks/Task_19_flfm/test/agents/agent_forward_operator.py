import torch

def forward_operator(estimate, psf_fft):
    """
    Computes the forward projection of the current 3D estimate.
    Projected_Image = Sum_z ( Conv2D(Estimate_z, PSF_z) )
    
    Args:
        estimate: 3D Tensor [nz, ny, nx]
        psf_fft: 3D Tensor [nz, ny, nx/2+1] (FFT of PSF)
    Returns:
        proj_image: 2D Tensor [ny, nx]
    """
    # 1. Transform estimate to frequency domain
    est_fft = torch.fft.rfft2(estimate, dim=(-2, -1))
    
    # 2. Convolve via multiplication in frequency domain
    product = est_fft * psf_fft
    
    # 3. Transform back to spatial domain
    # Note: s=estimate.shape[-2:] ensures correct output size for odd/even dimensions
    layers = torch.fft.irfft2(product, dim=(-2, -1), s=estimate.shape[-2:])
    
    # 4. Sum over the Z (depth) dimension to create the 2D projection
    proj_image = layers.sum(dim=0)
    
    return proj_image