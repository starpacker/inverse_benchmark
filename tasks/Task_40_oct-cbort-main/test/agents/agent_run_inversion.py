import numpy as np

def run_inversion(tomograms, settings):
    """
    Converts complex tomograms into a structural intensity image.
    Performs magnitude calculation, log compression, and contrast scaling.
    """
    tom1, tom2 = tomograms
    
    # Intensity (Structure)
    i1 = np.abs(tom1)**2
    i2 = np.abs(tom2)**2
    struct = i1 + i2
    
    # Log Compression (using log10 for better range fit)
    struct = 10 * np.log10(np.maximum(struct, 1e-10))
    
    # Contrast Scaling
    low, high = settings['contrastLowHigh']
    struct = (struct - low) / (high - low)
    struct = np.clip(struct, 0, 1)
    
    # Inversion
    if settings['invertGray']:
        struct = 1 - struct
        
    return struct
