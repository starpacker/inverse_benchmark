import numpy as np

import matplotlib

matplotlib.use('Agg')

def dim_match(A_shape, B_shape):
    A_out_shape = A_shape
    B_out_shape = B_shape
    if len(A_shape) < len(B_shape):
        for _ in range(len(A_shape), len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape), len(A_shape)):
            B_out_shape += (1,)
    return A_out_shape, B_out_shape

def forward_operator(image, mask, sensitivity_maps):
    """
    Implements A(x): SENSE -> FFT -> Mask
    image: 2D complex image
    mask: 2D mask
    sensitivity_maps: 3D coil maps (nx, ny, nc)
    Returns: Undersampled multicoil k-space
    """
    # 1. Apply SENSE (Sensitivity Encoding)
    # Equivalent to espirit.forward(image)
    sens_out_shape, im_out_shape = dim_match(sensitivity_maps.shape, image.shape)
    im_coils = np.multiply(image.reshape(im_out_shape), sensitivity_maps.reshape(sens_out_shape))
    
    # 2. Apply FFT
    # Equivalent to FFT2d.forward (embedded in FFT2d_kmask)
    axes = (0, 1)
    im_coils_shifted = np.fft.fftshift(im_coils, axes)
    ksp_full = np.fft.fft2(im_coils_shifted, s=None, axes=axes)
    ksp_full = np.fft.ifftshift(ksp_full, axes)
    
    # 3. Apply Mask
    # Equivalent to FFT2d_kmask.forward
    if len(ksp_full.shape) != len(mask.shape):
        ksp_out_shape, mask_out_shape = dim_match(ksp_full.shape, mask.shape)
        ksp_masked = np.multiply(ksp_full.reshape(ksp_out_shape), mask.reshape(mask_out_shape))
    else:
        ksp_masked = np.multiply(ksp_full, mask)
        
    return ksp_masked
