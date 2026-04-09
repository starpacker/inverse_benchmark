import numpy as np

from scipy.signal import convolve

def forward_operator(ground_truth, Psf, n_subarrays=5, signal_level=300):
    """
    Generates ISM data by convolving ground truth with PSF and adding noise.
    """
    print("Generating ISM Data (Forward Model)...")
    Nz, Nx, _ = ground_truth.shape
    blurred_phantom = np.empty([Nz, Nx, Nx, n_subarrays**2])
    
    # Apply signal level
    ground_truth_scaled = ground_truth * signal_level

    # Convolve
    for i in range(n_subarrays**2):
        for j in range(Nz):
            blurred_phantom[j, :, :, i] = convolve(ground_truth_scaled[j], Psf[j, :, :, i], mode='same')

    dataset_t = np.uint16(blurred_phantom.sum(axis=0))
    data_ISM_noise = np.random.poisson(dataset_t) # Adding Poisson noise
    
    return data_ISM_noise, ground_truth_scaled
