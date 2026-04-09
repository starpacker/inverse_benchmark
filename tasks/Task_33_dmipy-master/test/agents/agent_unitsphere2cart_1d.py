import numpy as np

def unitsphere2cart_1d(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to cartesian (x, y, z).
    """
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])
