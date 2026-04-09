import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

from skimage.data import shepp_logan_phantom

def load_and_preprocess_data(vol_size=64):
    """
    Generate a 3D phantom from skimage's 2D Shepp-Logan,
    with z-dependent scaling to simulate a 3D ellipsoidal volume.
    The phantom is zero outside the inscribed circle (required for
    radon(..., circle=True)).
    
    Returns:
        phantom: 3D numpy array of shape (vol_size, vol_size, vol_size)
    """
    N = vol_size
    # Get 2D Shepp-Logan and resize
    sl2d = shepp_logan_phantom()
    sl2d = resize(sl2d, (N, N), anti_aliasing=True).astype(np.float64)

    # Ensure zero outside inscribed circle
    yy, xx = np.ogrid[-1:1:complex(N), -1:1:complex(N)]
    circle = xx ** 2 + yy ** 2 <= 1.0

    phantom = np.zeros((N, N, N), dtype=np.float64)
    for iz in range(N):
        z = 2.0 * iz / (N - 1) - 1.0
        z_env = max(1.0 - z ** 2 * 1.5, 0.0)
        if z_env < 0.01:
            continue
        # Scale the 2D phantom by z-envelope
        s = sl2d * z_env
        s[~circle] = 0.0
        phantom[iz] = s

    # Normalize to [0, 1]
    vmin, vmax = phantom.min(), phantom.max()
    if vmax > vmin:
        phantom = (phantom - vmin) / (vmax - vmin)
    phantom = np.clip(phantom, 0, 1)
    return phantom
