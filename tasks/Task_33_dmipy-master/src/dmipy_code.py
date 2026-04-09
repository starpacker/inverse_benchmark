import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# =============================================================================
# Helper Functions (Geometry & Math)
# =============================================================================

def unitsphere2cart_1d(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to cartesian (x, y, z).
    """
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def fibonacci_sphere(samples=60):
    """
    Generates points distributed on a sphere using the Fibonacci spiral.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

# =============================================================================
# 1. Load and Preprocess Data
# =============================================================================

def load_and_preprocess_data(snr=30):
    """
    Generates synthetic Diffusion MRI data (simulating a 'load' process).
    Constructs a multi-shell acquisition scheme and generates noisy signal 
    based on a ground truth Ball & Stick model.

    Returns:
        tuple: (bvalues, gradient_directions, signal_noisy, gt_params)
    """
    # 1. Create Acquisition Scheme (b=0, 1000, 2000)
    # Generate 30 directions per shell
    bvecs_shell = fibonacci_sphere(30)
    
    # b-values: 5 b0s, 30 b1000, 30 b2000
    bvalues = np.concatenate([
        np.zeros(5),
        np.ones(30) * 1000e6,
        np.ones(30) * 2000e6
    ])
    
    # b-vectors: Stack shells
    gradient_directions = np.concatenate([
        np.zeros((5, 3)), # b0
        bvecs_shell,
        bvecs_shell
    ])
    # Set b0 vectors to x-axis to avoid NaNs in normalization, though magnitude is 0
    gradient_directions[0:5] = [1.0, 0.0, 0.0]
    
    # Normalize gradient directions
    norms = np.linalg.norm(gradient_directions, axis=1)
    norms[norms == 0] = 1.0
    gradient_directions = gradient_directions / norms[:, None]

    # 2. Define Ground Truth Parameters
    # f_stick, theta, phi, lambda_par, lambda_iso
    gt_f_stick = 0.6
    gt_theta = np.pi / 3
    gt_phi = np.pi / 4
    gt_lambda_par = 1.7e-9  # 1.7 um^2/ms
    gt_lambda_iso = 3.0e-9  # 3.0 um^2/ms
    
    gt_params = np.array([gt_f_stick, gt_theta, gt_phi, gt_lambda_par, gt_lambda_iso])

    # 3. Generate Noiseless Signal using the Forward Operator
    # We call the forward operator defined later, but since Python is dynamic, 
    # we can conceptually use the logic here or ensure order of execution. 
    # For this function, we explicitly implement the generation logic to be self-contained.
    
    # --- Generation Logic Start ---
    mu_cart = unitsphere2cart_1d(gt_theta, gt_phi)
    dot_prod = np.dot(gradient_directions, mu_cart)
    
    # Stick component: E = exp(-b * lambda_par * (n . mu)^2)
    E_stick = np.exp(-bvalues * gt_lambda_par * dot_prod**2)
    
    # Ball component: E = exp(-b * lambda_iso)
    E_ball = np.exp(-bvalues * gt_lambda_iso)
    
    signal_noiseless = gt_f_stick * E_stick + (1 - gt_f_stick) * E_ball
    # --- Generation Logic End ---

    # 4. Add Rician Noise
    # Signal amplitude assumed ~1.0 for b0
    sigma = 1.0 / snr
    noise_r = np.random.normal(0, sigma, signal_noiseless.shape)
    noise_i = np.random.normal(0, sigma, signal_noiseless.shape)
    signal_noisy = np.sqrt((signal_noiseless + noise_r)**2 + noise_i**2)
    
    return bvalues, gradient_directions, signal_noisy, gt_params

# =============================================================================
# 2. Forward Operator
# =============================================================================

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

# =============================================================================
# 3. Run Inversion
# =============================================================================

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

# =============================================================================
# 4. Evaluate Results
# =============================================================================

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

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # 1. Load Data
    bvals, bvecs, signal_data, gt_params = load_and_preprocess_data(snr=30)
    
    # 2. (Forward operator is called implicitly during inversion and evaluation)
    
    # 3. Run Inversion
    print("Running Inversion (Ball & Stick Model)...")
    fitted_params = run_inversion(bvals, bvecs, signal_data)
    
    # 4. Evaluate
    evaluate_results(fitted_params, gt_params, bvals, bvecs, signal_data)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")