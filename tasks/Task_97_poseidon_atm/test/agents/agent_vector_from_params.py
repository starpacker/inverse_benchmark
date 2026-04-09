import numpy as np

import matplotlib

matplotlib.use('Agg')

def vector_from_params(params, r_jup):
    """
    Convert parameter dict to vector.
    """
    return np.array([
        params["T"],
        params["log_X_H2O"],
        params["log_X_CH4"],
        params["log_X_CO2"],
        params["R_p"] / r_jup,
    ])
