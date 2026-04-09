import numpy as np

import matplotlib

matplotlib.use("Agg")

def effective_temperature(log_age, metallicity):
    """Simple mapping from age+Z to an effective temperature for the
    composite stellar population.  This is a toy analytic approximation:
      T_eff ~ 5000 * (age/1e9)^{-0.15} * (Z/0.02)^{0.05}  K
    Young, metal-poor populations are hotter.
    """
    age_gyr = 10.0 ** (log_age - 9.0)  # age in Gyr
    T = 5500.0 * age_gyr ** (-0.18) * (metallicity / 0.02) ** 0.05
    return np.clip(T, 2500.0, 50000.0)
