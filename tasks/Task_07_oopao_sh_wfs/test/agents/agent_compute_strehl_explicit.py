import numpy as np


# --- Extracted Dependencies ---

def compute_strehl_explicit(psf, psf_ref):
    """
    Computes Strehl Ratio explicitly using OTF (Optical Transfer Function) method.
    Strehl = Peak(PSF) / Peak(PSF_perfect)
           ~ Sum(OTF) / Sum(OTF_perfect)
    """
    otf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    otf_ref = np.abs(np.fft.fftshift(np.fft.fft2(psf_ref)))
    
    strehl = np.sum(otf) / np.sum(otf_ref)
    
    return strehl * 100
