import numpy as np

def unitsphere2cart_1d(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to cartesian (x, y, z).
    """
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def forward_operator(params, bvalues, gradient_directions):
    """
    Computes the diffusion signal for the Ball & Stick model.
    
    Args:
        params: array-like [f_stick, theta, phi, lambda_par, lambda_iso]
                Note: These must be PHYSICAL units (SI).
        bvalues: array (N,)
        gradient_directions: array (N, 3) normalized
        
    Returns:
        y_pred: array (N,) predicted signal attenuation.
    """
    f_stick = params[0]
    theta = params[1]
    phi = params[2]
    lambda_par = params[3]
    lambda_iso = params[4]
    
    # Convert orientation angles to Cartesian vector
    mu_cart = unitsphere2cart_1d(theta, phi)
    
    # Calculate Stick Component (C1Stick)
    # Model: E = exp(-b * lambda_par * (n . mu)^2)
    dot_prod = np.dot(gradient_directions, mu_cart)
    E_stick = np.exp(-bvalues * lambda_par * dot_prod**2)
    
    # Calculate Ball Component (G1Ball)
    # Model: E = exp(-b * lambda_iso)
    E_ball = np.exp(-bvalues * lambda_iso)
    
    # Combine Components
    # Signal = f * Stick + (1-f) * Ball
    y_pred = f_stick * E_stick + (1.0 - f_stick) * E_ball
    
    return y_pred
