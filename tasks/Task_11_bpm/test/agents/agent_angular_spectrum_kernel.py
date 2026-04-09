import torch


# --- Extracted Dependencies ---

def angular_spectrum_kernel(domain_size, spec_pixel_size, pixel_size, km):
    assert domain_size[1] % 2 == 0 and domain_size[2] % 2 == 0, "domain_size[1] and domain_size[2] must be even"
    kx = (torch.linspace((-domain_size[1] // 2 + 1), (domain_size[1] // 2), domain_size[1]) - 1) * spec_pixel_size
    ky = (torch.linspace((-domain_size[2] // 2 + 1), (domain_size[2] // 2), domain_size[2]) - 1) * spec_pixel_size
    [Ky, Kx] = torch.meshgrid(ky, kx, indexing='ij')
    K2 = Kx**2 + Ky**2
    Kz = torch.sqrt(-K2 + km**2 + 0j)
    Kz[-K2 + km**2 < 0] = 0.
    kernel = torch.exp(1j * Kz * pixel_size[0])
    ol_correction = km / Kz
    ol_correction[-K2 + (km * 1.2 / 1.33)**2 < 0] = 0.
    return torch.fft.fftshift(kernel), torch.fft.fftshift(ol_correction)
