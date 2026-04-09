import numpy as np
from PIL import Image

def load_image(fp, dtype="float32"):
    """Load an image file and return as numpy array."""
    img = Image.open(fp)
    img_array = np.array(img).astype(dtype)
    return img_array

def normalize_image(img):
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def downsample_image(img, factor):
    """Downsample image by a given factor using simple slicing."""
    if factor == 1:
        return img
    if len(img.shape) == 2:
        return img[::factor, ::factor]
    elif len(img.shape) == 3:
        return img[::factor, ::factor, :]
    else:
        return img

def prepare_4d_array(img):
    """Ensure image is 4D: (depth, height, width, channels)."""
    if len(img.shape) == 2:
        # Add channel dimension: (H, W) -> (H, W, 1)
        img = img[:, :, np.newaxis]
    if len(img.shape) == 3:
        # Add depth/batch dimension: (H, W, C) -> (1, H, W, C)
        img = img[np.newaxis, :, :, :]
    return img

def load_and_preprocess_data(psf_path, data_path, downsample=4):
    """
    Load PSF and measurement data, preprocess them.
    
    Args:
        psf_path: Path to PSF image file
        data_path: Path to measurement data file
        downsample: Downsampling factor
    
    Returns:
        Dictionary containing preprocessed PSF and measurement data
    """
    print(f"Loading data from {data_path}...")
    print(f"Loading PSF from {psf_path}...")
    
    psf_raw = load_image(psf_path, dtype="float32")
    data_raw = load_image(data_path, dtype="float32")
    
    psf_ds = downsample_image(psf_raw, downsample)
    data_ds = downsample_image(data_raw, downsample)
    
    psf_norm = normalize_image(psf_ds)
    data_norm = normalize_image(data_ds)
    
    psf = prepare_4d_array(psf_norm)
    data = prepare_4d_array(data_norm)
    
    print(f"Data shape: {data.shape}")
    print(f"PSF shape: {psf.shape}")
    
    return {"psf": psf, "data": data}