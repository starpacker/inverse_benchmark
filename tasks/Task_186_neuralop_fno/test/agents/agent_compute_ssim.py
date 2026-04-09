import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_ssim(ref, test):
    """Compute SSIM."""
    from skimage.metrics import structural_similarity as ssim
    data_range = ref.max() - ref.min()
    if data_range == 0:
        data_range = 1.0
    return ssim(ref, test, data_range=data_range)
