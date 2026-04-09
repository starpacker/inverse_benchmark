import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

GT_DENSITY = 0.5

GT_CENTER = [0.0, 0.0, -150.0]

GT_RADIUS = 100.0

def create_ground_truth(mesh_info):
    """Create ground truth density model: spherical anomaly."""
    cc = mesh_info['cell_centers']
    dist = np.sqrt(
        (cc[:, 0] - GT_CENTER[0]) ** 2 +
        (cc[:, 1] - GT_CENTER[1]) ** 2 +
        (cc[:, 2] - GT_CENTER[2]) ** 2
    )
    model_gt = np.zeros(mesh_info['n_cells'])
    model_gt[dist < GT_RADIUS] = GT_DENSITY

    # Add a smaller secondary anomaly
    dist2 = np.sqrt(
        (cc[:, 0] - (GT_CENTER[0] + 200)) ** 2 +
        (cc[:, 1] - (GT_CENTER[1] - 150)) ** 2 +
        (cc[:, 2] - (GT_CENTER[2] - 50)) ** 2
    )
    model_gt[dist2 < GT_RADIUS * 0.6] = -0.3

    return model_gt
