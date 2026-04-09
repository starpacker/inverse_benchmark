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

def evaluate_results(fitted_params, gt_params, bvalues, gradient_directions, data):
    """
    Compares fitted parameters with ground truth and calculates reconstruction error.
    """
    print("\n5. Evaluation Results:")
    print("---------------------------------------------------------------")
    print(f"{'PARAMETER':<12} | {'GROUND TRUTH':<12} | {'ESTIMATED':<12} | {'ERROR':<12}")
    print("---------------------------------------------------------------")
    
    names = ["f_stick", "theta (rad)", "phi (rad)", "D_par (m2/s)", "D_iso (m2/s)"]
    
    for i, name in enumerate(names):
        gt = gt_params[i]
        est = fitted_params[i]
        err = np.abs(gt - est)
        print(f"{name:<12} | {gt:<12.4g} | {est:<12.4g} | {err:<12.4g}")
        
    print("---------------------------------------------------------------")
    
    # Calculate Signal Reconstruction Error
    y_est = forward_operator(fitted_params, bvalues, gradient_directions)
    mse = np.mean((data - y_est)**2)
    
    # PSNR calculation (assuming peak signal is ~1.0)
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')
        
    print(f"\nSignal Reconstruction PSNR: {psnr:.2f} dB")
    print(f"Signal MSE: {mse:.2e}")
    
    return mse
