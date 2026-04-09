import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

N_CELLS_X = 20

N_CELLS_Y = 20

CELL_SIZE_X = 50.0

CELL_SIZE_Y = 50.0

N_RX_X = 15

N_RX_Y = 15

RX_HEIGHT = 1.0

def create_receiver_locations():
    """Create surface gravity receiver locations."""
    rx_x = np.linspace(
        -N_CELLS_X * CELL_SIZE_X / 2 * 0.7,
        N_CELLS_X * CELL_SIZE_X / 2 * 0.7,
        N_RX_X
    )
    rx_y = np.linspace(
        -N_CELLS_Y * CELL_SIZE_Y / 2 * 0.7,
        N_CELLS_Y * CELL_SIZE_Y / 2 * 0.7,
        N_RX_Y
    )
    rx_xx, rx_yy = np.meshgrid(rx_x, rx_y)
    rx_locs = np.c_[
        rx_xx.ravel(),
        rx_yy.ravel(),
        np.full(N_RX_X * N_RX_Y, RX_HEIGHT)
    ]
    return rx_locs
