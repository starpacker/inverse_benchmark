import numpy as np


# --- Extracted Dependencies ---

def integral_controller_explicit(current_command, slopes, reconstructor, gain):
    """
    Explicit Implementation of Integral Controller:
    u[k] = u[k-1] - g * R * s[k]
    """
    delta_command = np.matmul(reconstructor, slopes)
    next_command = current_command - gain * delta_command
    
    return next_command
