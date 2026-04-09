import torch
import torch.nn.functional as F

def forward_operator(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, mag):
    """
    Forward model: compute sub-spectrum intensities from complex image.
    
    Args:
        img_complex (torch.Tensor): High-res complex object (Batch, H, W)
        led_num (list/range): List of LED indices to process
        x_0, y_0, x_1, y_1 (list/Tensor): Crop coordinates for each LED
        spectrum_mask (torch.Tensor): Pupil function (Batch, h, w)
        mag (int): Magnification/Upsampling factor
        
    Returns:
        torch.Tensor: Amplitude of low-res images (Batch, N_LED, h, w)
    """
    # 1. FFT of complex image
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    
    # 2. Pad to high-res spectrum grid
    # Calculate padding required to match the effective high-res field of view
    to_pad_x = (spectrum_mask.shape[-2] * mag - O.shape[-2]) // 2
    to_pad_y = (spectrum_mask.shape[-1] * mag - O.shape[-1]) // 2
    
    # Pad format: (left, right, top, bottom, front, back)
    # We pad the last two dimensions (H, W)
    O = F.pad(O, (to_pad_y, to_pad_y, to_pad_x, to_pad_x, 0, 0), "constant", 0)

    # 3. Extract sub-apertures for each LED
    # Stack crops along dim=1 (the LED/channel dimension)
    O_sub = torch.stack(
        [O[:, x_0[i]:x_1[i], y_0[i]:y_1[i]] for i in range(len(led_num))], dim=1
    )
    
    # 4. Apply pupil mask
    O_sub = O_sub * spectrum_mask
    
    # 5. IFFT and compute amplitude (magnitude)
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub)

    return oI_sub