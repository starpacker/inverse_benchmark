import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def photoelectric_cross_section(E_keV):
    """Approximate photoelectric absorption cross-section in cm^2.
    Simplified Morrison & McCammon approximation."""
    return 2.0e-22 * (E_keV) ** (-8.0/3.0)
