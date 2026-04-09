import numpy as np

import matplotlib

matplotlib.use("Agg")

def radiation_P(strike_deg, dip_deg, rake_deg, azimuth_deg, takeoff_deg):
    """
    P-wave radiation pattern for a double-couple source.
    Aki & Richards (2002), Eq. 4.29.
    """
    s = np.radians(strike_deg)
    d = np.radians(dip_deg)
    r = np.radians(rake_deg)
    az = np.radians(azimuth_deg)
    ih = np.radians(takeoff_deg)
    phi = az - s

    R = (np.cos(r) * np.sin(d) * np.sin(ih)**2 * np.sin(2 * phi)
         - np.cos(r) * np.cos(d) * np.sin(2 * ih) * np.cos(phi)
         + np.sin(r) * np.sin(2 * d) * (np.cos(ih)**2 - np.sin(ih)**2 * np.sin(phi)**2)
         + np.sin(r) * np.cos(2 * d) * np.sin(2 * ih) * np.sin(phi))
    return R
