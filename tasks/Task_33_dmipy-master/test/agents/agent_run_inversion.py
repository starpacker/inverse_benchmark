import numpy as np

import scipy.optimize as opt

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

def run_inversion(bvalues, gradient_directions, data):
    """
    Performs Non-Linear Least Squares (NLLS) fitting to recover model parameters.
    
    Args:
        bvalues: Acquisition b-values
        gradient_directions: Acquisition gradients
        data: Observed noisy signal
        
    Returns:
        fitted_params_physical: array [f_stick, theta, phi, lambda_par, lambda_iso]
    """
    
    # Scaling factor to make diffusivities order of magnitude ~1.0 for the optimizer
    diff_scale = 1e-9
    
    # Wrapper for objective function handling scaling
    def objective_function(x):
        # Unpack optimizer parameters
        # x = [f, theta, phi, d_par_scaled, d_iso_scaled]
        
        # Reconstruct physical parameters
        params_physical = np.array([
            x[0],           # f_stick
            x[1],           # theta
            x[2],           # phi
            x[3] * diff_scale, # lambda_par
            x[4] * diff_scale  # lambda_iso
        ])
        
        # Forward pass
        y_pred = forward_operator(params_physical, bvalues, gradient_directions)
        
        # Sum of Squared Errors
        return np.sum((data - y_pred)**2)

    # Initial Guess (x0)
    # f=0.5, theta=0, phi=0, D_par=1.7 (scaled), D_iso=3.0 (scaled)
    x0 = [0.5, 0.0, 0.0, 1.7, 3.0]
    
    # Bounds for the optimizer
    # f: [0, 1], theta: [0, pi], phi: [-pi, pi], D: [0.1, 3.5] (scaled)
    bounds = [
        (0.0, 1.0),
        (0.0, np.pi),
        (-np.pi, np.pi),
        (0.1, 3.5),
        (0.1, 3.5)
    ]
    
    # Run Optimization
    res = opt.minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        bounds=bounds
    )
    
    # Convert result back to physical units
    fitted_params_physical = np.array([
        res.x[0],
        res.x[1],
        res.x[2],
        res.x[3] * diff_scale,
        res.x[4] * diff_scale
    ])
    
    return fitted_params_physical
