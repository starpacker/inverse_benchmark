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
