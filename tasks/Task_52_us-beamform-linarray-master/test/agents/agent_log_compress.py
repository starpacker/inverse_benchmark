import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_compress(data, dynamic_range, reject_level, bright_gain):
    """Log compression"""
    # Avoid log(0)
    data_safe = data.copy()
    data_safe[data_safe < 0] = 0
    xd_b = 20 * np.log10(data_safe + 1)
    xd_b2 = xd_b - np.max(xd_b)
    xd_b3 = xd_b2 + dynamic_range
    xd_b3[xd_b3 < 0] = 0
    xd_b3[xd_b3 <= reject_level] = 0
    xd_b3 = xd_b3 + bright_gain
    xd_b3[xd_b3 > dynamic_range] = dynamic_range
    return xd_b3
