import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_displacement_fields(height, width, n_frames=10):
    """Generate smooth spatially-varying sinusoidal displacement fields."""
    Y, X = np.meshgrid(np.arange(height, dtype=np.float64),
                        np.arange(width, dtype=np.float64), indexing='ij')

    sigma_spatial = min(height, width) / 2.0
    envelope = np.exp(-((X - width / 2)**2 + (Y - height / 2)**2) /
                       (2 * sigma_spatial**2))

    dx_fields = np.zeros((n_frames, height, width))
    dy_fields = np.zeros((n_frames, height, width))

    for t in range(n_frames):
        phase = 2 * np.pi * t / n_frames
        amp_x = 2.5 * np.sin(phase)
        amp_y = 1.8 * np.cos(phase)
        dx_fields[t] = amp_x * envelope
        dy_fields[t] = amp_y * envelope

    return dx_fields, dy_fields
