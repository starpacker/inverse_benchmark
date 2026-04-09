import matplotlib

matplotlib.use('Agg')

def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
