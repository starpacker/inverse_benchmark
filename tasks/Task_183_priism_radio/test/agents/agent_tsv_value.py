import numpy as np

import matplotlib

matplotlib.use('Agg')

def tsv_value(image):
    """Total Squared Variation: sum of squared differences."""
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    return np.sum(dx ** 2) + np.sum(dy ** 2)
