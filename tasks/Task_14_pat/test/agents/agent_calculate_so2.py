import numpy as np


# --- Extracted Dependencies ---

def calculate_so2(concentrations):
    """
    Calculate oxygen saturation sO2 = HbO2 / (Hb + HbO2).
    
    Args:
        concentrations: Array of shape (2, nz, ny, nx)
        
    Returns:
        so2: Oxygen saturation map, shape (nz, ny, nx)
    """
    hb = concentrations[0]
    hbo2 = concentrations[1]
    
    total_hb = hb + hbo2
    mask = total_hb > (0.1 * np.max(total_hb))
    
    so2 = np.zeros_like(hbo2)
    so2[mask] = hbo2[mask] / total_hb[mask]
    
    so2 = np.clip(so2, 0, 1)
    
    return so2
