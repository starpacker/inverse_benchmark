import numpy as np

def boundary_attenuation(n_incidence, n_transmission=1.0):
    """Calculate the boundary attenuation factor A using Fresnel's law (Robin BC)."""
    n = n_incidence / n_transmission
    R0 = ((n - 1.) ** 2) / ((n + 1.) ** 2)
    theta_c = np.arcsin(1.0 / n)
    cos_theta_c = np.cos(theta_c)
    A = (2.0 / (1.0 - R0) - 1.0 + np.abs(cos_theta_c) ** 3) / (1.0 - np.abs(cos_theta_c) ** 2)
    return A
