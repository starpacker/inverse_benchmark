import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_scan_positions(obj_size, probe_size, overlap):
    """Regular grid scan with given overlap fraction."""
    step = max(int(probe_size * (1 - overlap)), 1)
    pos_1d = np.arange(0, obj_size - probe_size + 1, step)
    if len(pos_1d) == 0:
        pos_1d = np.array([0])
    if pos_1d[-1] < obj_size - probe_size:
        pos_1d = np.append(pos_1d, obj_size - probe_size)
    return [(int(py), int(px)) for py in pos_1d for px in pos_1d]
