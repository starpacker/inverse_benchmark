import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_norm(refs, true_a, wavelength, tt_min, tt_max):
    """Find raw integrated intensity of strongest peak for normalization."""
    max_raw = 0
    for q2, F2, mult in refs:
        d = true_a / np.sqrt(q2)
        s = wavelength / (2*d)
        if abs(s) >= 1:
            continue
        tt = 2*np.degrees(np.arcsin(s))
        if tt < tt_min or tt > tt_max:
            continue
        th = np.radians(tt/2)
        lp = (1 + np.cos(2*th)**2) / (np.sin(th)**2 * np.cos(th))
        dw = np.exp(-0.5*(np.sin(th)/wavelength)**2)
        raw = mult * F2 * lp * dw
        if raw > max_raw:
            max_raw = raw
    return max_raw
