import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the target function to the Python path
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Error: Could not import 'run_inversion' from 'agent_run_inversion.py'")
    sys.exit(1)

# --- INJECTING REFEREE / EVALUATION LOGIC (Reference B) ---

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
    
    # Handle cases where ground truth might not be explicitly passed or might be the "standard output"
    # If gt_params is None or has wrong shape, we might skip parameter comparison and rely on MSE
    gt_available = False
    if gt_params is not None and len(gt_params) == len(names):
        gt_available = True
        for i, name in enumerate(names):
            gt = gt_params[i]
            est = fitted_params[i]
            err = np.abs(gt - est)
            print(f"{name:<12} | {gt:<12.4g} | {est:<12.4g} | {err:<12.4g}")
    else:
        print("Ground Truth parameters not available for direct comparison (using Reconstruction Error only).")
        
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

# --- MAIN VALIDATION LOGIC ---

def main():
    data_paths = ['/data/yjh/dmipy-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Filter for the relevant data file
    outer_data_path = None
    for path in data_paths:
        if 'standard_data_run_inversion.pkl' in path:
            outer_data_path = path
            break
            
    if not outer_data_path:
        print("Error: Standard data file not found.")
        sys.exit(1)
        
    print(f"Loading data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)
        
    # Extract arguments
    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    std_output = data_payload.get('output', None)
    
    # Context extraction for evaluation
    # Based on function signature: run_inversion(bvalues, gradient_directions, data)
    if len(args) >= 3:
        bvalues = args[0]
        gradient_directions = args[1]
        observed_data = args[2]
    elif 'bvalues' in kwargs and 'gradient_directions' in kwargs and 'data' in kwargs:
        bvalues = kwargs['bvalues']
        gradient_directions = kwargs['gradient_directions']
        observed_data = kwargs['data']
    else:
        # Fallback if args are mixed or partial (simplified assumption)
        print("Warning: Could not strictly parse input arguments for evaluation context. Attempting by index.")
        try:
            bvalues = args[0]
            gradient_directions = args[1]
            observed_data = args[2]
        except IndexError:
            print("Error: Insufficient arguments to reconstruct context.")
            sys.exit(1)

    print("\n1. Running Agent 'run_inversion'...")
    try:
        agent_result = run_inversion(*args, **kwargs)
        print("   Agent execution successful.")
    except Exception:
        print("   Agent execution failed!")
        traceback.print_exc()
        sys.exit(1)

    print("\n2. Evaluating Standard Result (Ground Truth / Reference)...")
    # In this context, the standard output acts as the reference solution found by the original code.
    # While it's not strictly "Ground Truth" (which would be the hidden params used to generate the data),
    # it serves as the benchmark for "expected performance".
    # We will pass std_output as 'gt_params' to the evaluator for display purposes, 
    # but the primary metric is MSE against the observed data.
    mse_std = evaluate_results(std_output, std_output, bvalues, gradient_directions, observed_data)

    print("\n3. Evaluating Agent Result...")
    # Here, we compare the Agent's result against the observed data.
    # We pass std_output as reference parameters to see how close we got to the reference implementation.
    mse_agent = evaluate_results(agent_result, std_output, bvalues, gradient_directions, observed_data)

    print("\n4. Comparison Summary:")
    print(f"   Standard MSE: {mse_std:.6e}")
    print(f"   Agent MSE:    {mse_agent:.6e}")

    # Success Criteria
    # Since this is an optimization problem (Loss/MSE), Lower is Better.
    # We allow the agent to be slightly worse (higher MSE) within a margin, or better.
    # Margin: 10% tolerance (factor of 1.1)
    
    # Note: If MSE is extremely small (near zero), relative comparison might be unstable.
    # We add a small epsilon or check absolute threshold if both are very good.
    
    threshold_factor = 1.1
    if mse_std < 1e-9 and mse_agent < 1e-9:
        print("   Both results are extremely accurate (near zero error). Test Passed.")
        sys.exit(0)
        
    if mse_agent <= mse_std * threshold_factor:
        print(f"   Success: Agent MSE is within acceptable range ({threshold_factor}x of Standard).")
        sys.exit(0)
    else:
        print(f"   Failure: Agent MSE is significantly higher than Standard.")
        sys.exit(1)

if __name__ == "__main__":
    main()