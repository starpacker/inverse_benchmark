import numpy as np
import torch

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def load_and_preprocess_data(data_config, physics_config):
    device = get_device()
    
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    
    crop_size = data_config['crop_size']
    domain_size = data_config['domain_size']
    
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    
    try:
        u_in = np.load('data_folder/u_in.npy')
        u_in = torch.from_numpy(u_in).to(device)
        u_out = np.load('data_folder/u_out.npy')
        u_out = torch.from_numpy(u_out).to(device)
        k_scan_samp = np.load('data_folder/k_samp.npy')
    except FileNotFoundError:
        print("Error: Data files not found. Run generate_dummy_data() first.")
        raise
    
    temp = k_scan_samp * spec_pixel_size / km
    
    arg_sq = temp[:, 0] ** 2 + temp[:, 1] ** 2
    arg_sq = np.clip(arg_sq, 0, 1) 
    
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(arg_sq)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).float().to(device)
    
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    
    if crop_z is not None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    
    p_kernel, _ = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution, km)
    p_kernel = p_kernel.to(device)
    
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** bg_z))
    
    preprocessed_data = {
        'u_inlet': u_inlet,
        'u_outlet': u_outlet,
        'p_kernel': p_kernel,
        'bpm_cosFactor': bpm_cosFactor,
        'resolution': resolution,
        'domain_size': domain_size,
        'region_z': region_z,
        'k0': k0,
        'km': km,
        'device': device
    }
    
    return preprocessed_data