import numpy as np

import scipy.fft

import matplotlib.pyplot as plt

def minus_log(data):
    """
    Computes the minus log of the data: P = -log(data).
    """
    data = np.where(data <= 0, 1e-6, data)
    return -np.log(data)

def radon_transform_logic(image, theta):
    """
    Explicit implementation of the Radon Transform (Forward Projector).
    """
    num_angles = len(theta)
    N = image.shape[1] 
    sinogram = np.zeros((num_angles, N), dtype=np.float32)

    for i, angle in enumerate(theta):
        # Rotate the image. order=1 (linear)
        rotated = scipy.ndimage.rotate(image, -angle, reshape=False, order=1, mode='constant', cval=0.0)
        sinogram[i] = rotated.sum(axis=0)
        
    return sinogram

def load_and_preprocess_data(file_path, I0=100000.0, downsample_size=(256, 256)):
    """
    Loads image, simulates noise, and prepares sinogram.
    Returns: original_image, sinogram_noisy, theta
    """
    # Import logic restricted to function scope or global try/except used inside
    try:
        import tifffile
        read_tiff = tifffile.imread
    except ImportError:
        try:
            from skimage import io
            read_tiff = io.imread
        except ImportError:
            read_tiff = plt.imread

    print(f"Loading data from: {file_path}")
    original_image = read_tiff(file_path)
    original_image = original_image.astype(np.float32)
    
    if original_image.max() > 1.0:
        original_image /= original_image.max()
        
    # Downsample
    if original_image.shape[0] > 512:
        from skimage.transform import resize
        original_image = resize(original_image, downsample_size, anti_aliasing=True)
    
    # Simulate Acquisition
    theta = np.linspace(0, 180, 360, endpoint=False)
    sinogram_clean = radon_transform_logic(original_image, theta)
    
    # Add Poisson Noise
    transmission = I0 * np.exp(-sinogram_clean)
    transmission_noisy = np.random.poisson(transmission).astype(np.float32)
    
    # Preprocess (Normalization + Log)
    normalized_proj = transmission_noisy / I0
    sinogram_noisy = minus_log(normalized_proj)
    
    return original_image, sinogram_noisy, theta
