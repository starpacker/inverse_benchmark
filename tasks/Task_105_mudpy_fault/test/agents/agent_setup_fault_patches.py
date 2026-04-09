import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_105_mudpy_fault"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def setup_fault_patches(nx, ny, length, width, depth, dip_deg, strike_deg):
    """Discretize fault plane into rectangular patches."""
    dip_rad = np.deg2rad(dip_deg)
    strike_rad = np.deg2rad(strike_deg)

    dx = length / nx
    dy = width / ny

    patches = []
    for j in range(ny):
        for i in range(nx):
            cx = (i + 0.5) * dx - length / 2
            dip_dist = (j + 0.5) * dy
            cy_offset = dip_dist * np.cos(dip_rad)
            cz = depth + dip_dist * np.sin(dip_rad)

            cx_rot = cx * np.cos(strike_rad)
            cy_rot = cx * np.sin(strike_rad) + cy_offset

            patches.append({
                "cx": cx_rot,
                "cy": cy_rot,
                "depth": cz,
                "length": dx,
                "width": dy,
                "dip_rad": dip_rad,
                "i": i,
                "j": j,
            })

    return patches
