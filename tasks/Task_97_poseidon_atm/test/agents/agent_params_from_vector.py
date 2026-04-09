import matplotlib

matplotlib.use('Agg')

def params_from_vector(x, r_jup):
    """
    Convert parameter vector to dict.
    x = [T, log_X_H2O, log_X_CH4, log_X_CO2, R_p_factor]
    """
    return {
        "T":         x[0],
        "log_X_H2O": x[1],
        "log_X_CH4": x[2],
        "log_X_CO2": x[3],
        "R_p":       x[4] * r_jup,
    }
