import numpy as np

from PIL import Image

from skimage.transform import resize as sk_resize

import os

def custom_resize(img, factor):
    """
    Downsample image by a factor of 1/2^k using box filtering.
    """
    num = int(-np.log2(factor))
    for i in range(num):
        h, w = img.shape[:2]
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        img = img[:h_even, :w_even]
        img = 0.25 * (img[::2, ::2, ...] + img[1::2, ::2, ...] + img[::2, 1::2, ...] + img[1::2, 1::2, ...])
    return img

def load_and_preprocess_data(psf_path, data_path, gt_path, config):
    """
    Loads images, applies resizing, background subtraction, and normalization.
    """
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF not found at {psf_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    # Load PSF
    psf_img = Image.open(psf_path).convert('L')
    psf_raw = np.array(psf_img, dtype='float32')
    
    # Load Data
    data_img = Image.open(data_path).convert('L')
    data_raw = np.array(data_img, dtype='float32')

    # Load GT (Optional)
    gt_raw = None
    if gt_path and os.path.exists(gt_path):
        gt_img = Image.open(gt_path).convert('L') # Usually GT is color, convert to Gray for processing
        gt_raw = np.array(gt_img, dtype='float32')

    # Preprocess (Resize)
    f = config['f']
    psf = custom_resize(psf_raw, f)
    data = custom_resize(data_raw, f)
    
    # Background Subtraction (based on corner)
    bg = np.mean(psf[5:15, 5:15])
    psf -= bg
    data -= bg
    
    # Normalization
    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())

    # Process GT if exists
    gt = None
    if gt_raw is not None:
        sensor_size = psf.shape
        # Use skimage resize for GT to match sensor size exactly (as per tutorial intent)
        gt = sk_resize(gt_raw, sensor_size, anti_aliasing=True)
        gt /= np.linalg.norm(gt.ravel())

    return psf, data, gt
