import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def create_crescent_image(N, fov):
    """Create a crescent-shaped black hole shadow model."""
    x = np.linspace(-fov / 2, fov / 2, N)
    y = np.linspace(-fov / 2, fov / 2, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    r_ring = 40.0
    width = 12.0
    ring = np.exp(-0.5 * ((R - r_ring) / (width / 2.35)) ** 2)
    asym = 1.0 + 0.6 * np.cos(np.arctan2(Y, X) - np.pi)
    image = ring * asym
    shadow = 1.0 - np.exp(-0.5 * (R / (r_ring * 0.5)) ** 2)
    image *= shadow
    image = np.maximum(image, 0)
    if image.sum() > 0:
        image /= image.sum()
    return image
