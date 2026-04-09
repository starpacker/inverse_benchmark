import matplotlib

matplotlib.use('Agg')

import numpy as np

def create_test_image(size: int, seed: int) -> np.ndarray:
    """Create a synthetic test image with geometric shapes and textures."""
    img = np.zeros((size, size), dtype=np.float64)

    # Background gradient
    yy, xx = np.mgrid[0:size, 0:size]
    img += 0.15 * (xx / size) + 0.1 * (yy / size)

    # Circles
    for cx, cy, r, v in [(35, 35, 18, 0.9), (90, 40, 14, 0.7),
                          (60, 90, 20, 0.8), (100, 100, 10, 0.6)]:
        mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        img[mask] = v

    # Rectangle
    img[20:50, 70:110] = 0.5

    # Sinusoidal texture
    img += 0.08 * np.sin(2 * np.pi * xx / 16) * np.cos(2 * np.pi * yy / 20)

    # Small bright dots (stars)
    rng = np.random.RandomState(seed)
    for _ in range(15):
        px, py = rng.randint(5, size - 5, size=2)
        img[py - 1:py + 2, px - 1:px + 2] = 0.95

    img = np.clip(img, 0.0, 1.0)
    return img
