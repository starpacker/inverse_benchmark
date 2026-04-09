import numpy as np


# --- Extracted Dependencies ---

def _compute_strehl(psf, psf_ref):
    """
    Helper: Computes Strehl Ratio using OTF (Optical Transfer Function) method.
    Strehl ~ Sum(OTF) / Sum(OTF_perfect)
    """
    otf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    otf_ref = np.abs(np.fft.fftshift(np.fft.fft2(psf_ref)))
    strehl = np.sum(otf) / np.sum(otf_ref)
    return strehl * 100
