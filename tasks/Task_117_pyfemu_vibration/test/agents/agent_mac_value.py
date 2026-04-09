import numpy as np

import matplotlib

matplotlib.use("Agg")

def mac_value(phi_a, phi_b):
    """MAC between two mode shape vectors."""
    num = np.dot(phi_a, phi_b) ** 2
    den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b)
    return num / (den + 1e-30)
