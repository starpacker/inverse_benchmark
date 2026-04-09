import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

GRAV_CONST = 6.674e-3

def prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho):
    """
    Compute vertical gravity component gz for a rectangular prism.
    
    Uses the analytical formula from Blakely (1996) / Nagy (1966).
    
    Parameters
    ----------
    x1, x2 : float  Prism x bounds
    y1, y2 : float  Prism y bounds
    z1, z2 : float  Prism z bounds (z positive downward in convention, but we use z negative for depth)
    xp, yp, zp : float  Observation point coordinates
    rho : float  Density contrast [g/cm³]
    
    Returns
    -------
    gz : float  Vertical gravity component [mGal]
    """
    # Shift coordinates relative to observation point
    dx = [x1 - xp, x2 - xp]
    dy = [y1 - yp, y2 - yp]
    dz = [z1 - zp, z2 - zp]
    
    gz = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = dx[i]
                y = dy[j]
                z = dz[k]
                r = np.sqrt(x**2 + y**2 + z**2)
                
                # Avoid singularities
                r = max(r, 1e-10)
                
                # Sign for the sum
                sign = (-1) ** (i + j + k)
                
                # Compute terms
                term1 = 0.0
                term2 = 0.0
                term3 = 0.0
                
                # x * ln(y + r)
                if abs(y + r) > 1e-10:
                    term1 = x * np.log(y + r)
                
                # y * ln(x + r)
                if abs(x + r) > 1e-10:
                    term2 = y * np.log(x + r)
                
                # z * arctan(xy / (zr))
                denom = z * r
                if abs(denom) > 1e-10:
                    term3 = z * np.arctan2(x * y, denom)
                
                gz += sign * (term1 + term2 - term3)
    
    # Convert to mGal (G in appropriate units)
    # G = 6.674e-11 m³/(kg·s²), density in g/cm³ = 1000 kg/m³
    # 1 mGal = 1e-5 m/s²
    # gz = G * rho * integral, with rho in g/cm³
    # Factor: 6.674e-11 * 1000 * 1e5 = 6.674e-3
    gz *= GRAV_CONST * rho
    
    return gz
