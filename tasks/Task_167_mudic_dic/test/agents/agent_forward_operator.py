import matplotlib

matplotlib.use('Agg')

import os

import sys

import logging

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

sys.path.insert(0, REPO_DIR)

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(x, displacement_function, omega, amp):
    """
    Forward operator: Given material coordinates, compute the displacement field
    using the known harmonic bilateral deformation model.
    
    The forward model is:
        u_x(x, y) = amp * sin(omega * x) * sin(omega * y)
        u_y(x, y) = amp * sin(omega * x) * sin(omega * y)
    
    Parameters
    ----------
    x : tuple of ndarray
        (x_coords, y_coords) - material point coordinates in image space
    displacement_function : callable
        The muDIC displacement function (harmonic_bilat)
    omega : float
        Angular frequency in the coordinate system of x
    amp : float
        Amplitude in the coordinate system of x
    
    Returns
    -------
    y_pred : tuple of ndarray
        (u_x, u_y) - predicted displacement components
    """
    x_coords, y_coords = x
    
    # Apply the harmonic bilateral displacement function
    u_x, u_y = displacement_function(x_coords, y_coords, omega=omega, amp=amp)
    
    return (u_x, u_y)
