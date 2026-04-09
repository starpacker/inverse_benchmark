import matplotlib

matplotlib.use('Agg')

def forward_operator(T21, T_fg, noise):
    """
    Apply the forward model: y = T_21 + T_fg*1000 + noise
    
    All outputs are in milliKelvin (mK).
    
    Args:
        T21: 21cm signal in mK (n_freq x n_angle)
        T_fg: foreground in Kelvin (n_freq x n_angle)
        noise: noise realization in mK (n_freq x n_angle)
    
    Returns:
        y_pred: predicted observation in mK (n_freq x n_angle)
    """
    y_pred = T21 + T_fg * 1000.0 + noise
    return y_pred
