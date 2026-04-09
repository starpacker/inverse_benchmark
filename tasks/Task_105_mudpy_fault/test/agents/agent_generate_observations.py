import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_observations(n_obs, fault_length, fault_width, seed):
    """Generate observation station positions around the fault."""
    np.random.seed(seed)
    obs_x = np.random.uniform(-fault_length, fault_length, n_obs)
    obs_y = np.random.uniform(-fault_width * 0.5, fault_width * 2.0, n_obs)
    return np.column_stack([obs_x, obs_y])
