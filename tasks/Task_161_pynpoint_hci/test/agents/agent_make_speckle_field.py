import numpy as np

import matplotlib

matplotlib.use("Agg")

def make_speckle_field(size, rng, n_modes=25, amp=0.03):
    """
    Quasi-static speckle field as a sum of random sinusoidal modes.
    Fixed in the pupil plane → identical in every frame.
    Returns a zero-mean, amplitude-controlled field.
    """
    field = np.zeros((size, size))
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_modes):
        kx = rng.uniform(-0.3, 0.3)
        ky = rng.uniform(-0.3, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.3, 1.0)
        field += a * np.cos(2 * np.pi * (kx * xx + ky * yy) + phase)
    field -= field.mean()
    field *= amp / (field.std() + 1e-10)
    return field
