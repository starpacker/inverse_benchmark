import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_background(energy, a=500.0, b=2.5, c=0.1):
    """
    Generate Bremsstrahlung background: exponential + scatter.
    B(E) = a * exp(-b*E) + c
    """
    return a * np.exp(-b * energy) + c
