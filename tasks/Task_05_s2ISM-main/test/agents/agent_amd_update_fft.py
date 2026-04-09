import torch

from torch import real, einsum

from torch.fft import rfftn, irfftn, ifftshift

import string

def partial_convolution_rfft(kernel: torch.Tensor, volume: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                             axis: str = 'jk', fourier: tuple = (False, False), padding: list = None):
    
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]
        
    if fourier[0] == False:
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel
    
    if fourier[1] == False:
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume
    
    conv = einsum(f'{dim1},{dim2}->{dim3}', kernel_fft, volume_fft)

    conv = irfftn(conv, dim=axis_list[2], s=padding)
    conv = ifftshift(conv, dim=axis_list[2])
    conv = real(conv)

    return conv

def amd_update_fft(img, obj, psf_fft, psf_m_fft, eps: float):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alphabet = string.ascii_lowercase
    n_dim = psf_fft.ndim
   
    str_psf = alphabet[:n_dim]
    str_first = alphabet[:n_dim-1]
    str_second = alphabet[1:n_dim]
    axis_str = alphabet[1:n_dim-1]
    
    img_estimate = partial_convolution_rfft(psf_fft, obj, dim1=str_psf, dim2=str_first, axis=axis_str, fourier=(1,0)).sum(0)
    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)
    del img_estimate
    update = partial_convolution_rfft(psf_m_fft, fraction, dim1=str_psf, dim2=str_second, axis=axis_str, fourier=(1,0)).sum(-1)
    del fraction
    obj_new = obj * update

    return obj_new
