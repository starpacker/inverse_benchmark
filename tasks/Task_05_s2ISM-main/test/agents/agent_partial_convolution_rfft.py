import torch

from torch import real, einsum

from torch.fft import rfftn, irfftn, ifftshift

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
