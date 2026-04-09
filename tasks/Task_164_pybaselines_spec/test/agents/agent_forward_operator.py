import matplotlib

matplotlib.use('Agg')

import os

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, "repo")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

sys.path.insert(0, REPO_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(true_signal, true_baseline, noise):
    """
    Apply the forward model: measured = signal + baseline + noise.
    
    This represents the physical measurement process where the observed
    spectrum is a combination of the true signal, a baseline contribution,
    and measurement noise.
    
    Parameters
    ----------
    true_signal : np.ndarray
        Ground truth signal (peaks).
    true_baseline : np.ndarray
        Ground truth baseline (polynomial + fluorescence).
    noise : np.ndarray
        Noise component.
    
    Returns
    -------
    measured : np.ndarray
        Measured/observed spectrum.
    """
    measured = true_signal + true_baseline + noise
    return measured
