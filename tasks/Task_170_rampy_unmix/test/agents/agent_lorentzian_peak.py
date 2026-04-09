import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def lorentzian_peak(x, center, width, height):
    """Single Lorentzian peak."""
    return height / (1 + ((x - center) / width) ** 2)
