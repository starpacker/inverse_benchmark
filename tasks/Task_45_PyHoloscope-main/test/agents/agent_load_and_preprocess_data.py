import numpy as np

from PIL import Image

import cv2 as cv

def load_and_preprocess_data(holo_path, back_path=None, downsample=1.0, window_func=None, precision="single"):
    """
    Loads hologram and background images, applies background subtraction,
    normalization (optional), windowing, and downsampling.
    
    Returns:
        processed_img (np.ndarray): The complex or real preprocessed field.
        raw_holo (np.ndarray): The raw loaded hologram for visualization.
        raw_back (np.ndarray): The raw loaded background (or None).
    """
    # 1. Define Precision
    if precision == "double":
        dtype_complex = "complex128"
        dtype_real = "float64"
    else:
        dtype_complex = "complex64"
        dtype_real = "float32"

    # 2. Helper: Load Image
    def _load_image(filename):
        im = Image.open(filename)
        # Handle multi-frame tiff if necessary, though typical for inline is single frame
        if hasattr(im, 'n_frames') and im.n_frames > 1:
            im.seek(0) # Just take first frame for this simple pipeline
        return np.array(im)

    # 3. Load Data
    raw_holo = _load_image(holo_path)
    img = raw_holo.astype(dtype_real)
    
    raw_back = None
    background = None
    if back_path:
        raw_back = _load_image(back_path)
        background = raw_back.astype(dtype_real)

    # 4. Remove Background
    # Logic: If complex, (Amp - sqrt(Back)) * exp(j*Phase). If real, Img - Back.
    if background is not None:
        if background.shape != img.shape:
             # Basic resize if shapes mismatch (robustness)
             background = cv.resize(background, (img.shape[1], img.shape[0]))
        
        if np.iscomplexobj(img):
            img_amp = np.abs(img)
            img_phase = np.angle(img)
            # Ensure non-negative inside sqrt for background
            bg_sqrt = np.sqrt(np.maximum(background, 0))
            img = (img_amp - bg_sqrt) * np.exp(1j * img_phase)
        else:
            img = img - background

    # 5. Apply Window
    # Create a window if one isn't passed but a string name might be (not implemented here, assuming array passed)
    if window_func is not None:
        if window_func.shape != img.shape:
            # Resize window to match image
            window_resized = cv.resize(window_func, (img.shape[1], img.shape[0]))
        else:
            window_resized = window_func
            
        if np.iscomplexobj(img):
            img.real *= window_resized
            img.imag *= window_resized
        else:
            img *= window_resized

    # 6. Downsample
    if downsample != 1.0:
        new_w = int(img.shape[1] / downsample / 2) * 2
        new_h = int(img.shape[0] / downsample / 2) * 2
        if np.iscomplexobj(img):
            # Resize real and imag separately
            r = cv.resize(img.real, (new_w, new_h))
            i = cv.resize(img.imag, (new_w, new_h))
            img = r + 1j*i
        else:
            img = cv.resize(img, (new_w, new_h))

    # Cast final result to correct complex type for propagation
    img = img.astype(dtype_complex)
    
    return img, raw_holo, raw_back
