import matplotlib

matplotlib.use('Agg')

def forward_operator(mode_params, freqs, bg_params):
    """
    Forward model: Generate power spectrum from mode parameters.
    
    P(ν) = Σ_nl H_nl · L(ν; f_nl, Γ_nl) + B(ν)
    
    where L is a Lorentzian profile and B is the background.
    
    Args:
        mode_params: list of tuples [(freq, height, width), ...]
        freqs: ndarray of frequency values
        bg_params: tuple (A, nu_c, alpha, W) for Harvey-like background
        
    Returns:
        spectrum: ndarray, the predicted power spectrum
    """
    # Background: Harvey-like profile + white noise
    A, nu_c, alpha, W = bg_params
    spectrum = A / (1.0 + (freqs / nu_c) ** alpha) + W
    
    # Add each mode as a Lorentzian
    for f0, H, gamma in mode_params:
        spectrum = spectrum + H / (1.0 + ((freqs - f0) / gamma) ** 2)
    
    return spectrum
