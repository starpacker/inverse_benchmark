import scipy.io as sio
import numpy as np
import torch

def load_and_preprocess_data(sample, color, num_modes, device):
    """
    Load FPM data from .mat file and preprocess it.
    Returns all necessary data structures for the inversion.
    """
    # 1. Load Data
    if sample == 'Siemens':
        # In a real scenario, this file must exist. 
        # For unit testing, sio.loadmat should be mocked.
        data_struct = sio.loadmat(f"data/{sample}/{sample}_{color}.mat")
        MAGimg = 3
    else:
        raise ValueError("Only Siemens sample is supported in this demo.")
    
    # 2. Extract and Crop Images
    I = data_struct["I_low"].astype("float32")
    # Crop to num_modes x num_modes
    I = I[0:int(num_modes), 0:int(num_modes), :]

    M = I.shape[0]
    N = I.shape[1]
    ID_len = I.shape[2]
    
    # 3. Extract NA Calibration
    NAs = data_struct["na_calib"].astype("float32")
    NAx = NAs[:, 0]
    NAy = NAs[:, 1]
    
    # 4. Set Wavelength
    if color == "r":
        wavelength = 0.632
    elif color == "g":
        wavelength = 0.5126
    elif color == "b":
        wavelength = 0.471
    else:
        wavelength = 0.632
        
    k0 = 2 * np.pi / wavelength
    
    # 5. Physical Parameters
    mag = data_struct["mag"].astype("float32")
    pixel_size = data_struct["dpix_c"].astype("float32")
    D_pixel = pixel_size / mag
    NA = data_struct["na_cal"].astype("float32")
    kmax = NA * k0
    
    # 6. Define High-Res Grid Dimensions
    MM = int(M * MAGimg)
    NN = int(N * MAGimg)
    
    # 7. Create High-Res Frequency Grid
    Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
    Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
    Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)
    
    # 8. Sort LEDs by Illumination Angle
    u = -NAx
    v = -NAy
    NAillu = np.sqrt(u**2 + v**2)
    order = np.argsort(NAillu)
    u = u[order]
    v = v[order]
    
    # 9. Calculate LED Positions in Fourier Space
    ledpos_true = np.zeros((ID_len, 2), dtype=int)
    for idx in range(ID_len):
        Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
        ledpos_true[idx, 0] = np.argmin(Fx1_temp)
        Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
        ledpos_true[idx, 1] = np.argmin(Fy1_temp)
    
    # Reorder images and normalize
    Isum = I[:, :, order] / np.max(I)
    
    # 10. Angular Spectrum Kernel (kzz)
    kxx, kyy = np.meshgrid(
        np.linspace(-np.pi/D_pixel, np.pi/D_pixel, M),
        np.linspace(-np.pi/D_pixel, np.pi/D_pixel, N)
    )
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    
    # Complex sqrt handling
    term = (k0**2 - krr.astype("complex64") ** 2)
    kzz_ampli = mask_k * np.abs(np.sqrt(term))
    kzz_phase = np.angle(np.sqrt(term))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)
    
    # 11. Generate Pupil Support
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1
    
    # 12. Convert to Torch Tensors
    Pupil0 = torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)
    kzz = torch.from_numpy(kzz).to(device).unsqueeze(0)
    Isum = torch.from_numpy(Isum).to(device)
    
    data_dict = {
        'Isum': Isum,
        'Pupil0': Pupil0,
        'kzz': kzz,
        'ledpos_true': ledpos_true,
        'M': M,
        'N': N,
        'MM': MM,
        'NN': NN,
        'ID_len': ID_len,
        'MAGimg': MAGimg,
    }
    
    return data_dict