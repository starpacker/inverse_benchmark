import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch import to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

# Add the current directory to sys.path to import local modules
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # If verification_utils is missing, define a fallback or fail.
    # Assuming it exists based on instructions, but good to handle gracefully.
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

# Inject helper functions needed for dill loading or function execution if they rely on globals
# The gen_data_code defines unitsphere2cart_1d and forward_operator which evaluate_results calls.
# We must ensure they are available in the namespace if evaluate_results relies on them being global,
# or if dill needs them to deserialize data.

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
    """
    f_stick = params[0]
    theta = params[1]
    phi = params[2]
    lambda_par = params[3]
    lambda_iso = params[4]
    mu_cart = unitsphere2cart_1d(theta, phi)
    dot_prod = np.dot(gradient_directions, mu_cart)
    E_stick = np.exp(-bvalues * lambda_par * dot_prod ** 2)
    E_ball = np.exp(-bvalues * lambda_iso)
    y_pred = f_stick * E_stick + (1.0 - f_stick) * E_ball
    return y_pred

# Inject these into the global namespace where dill might look for them
# or where the imported function expects them (though usually imports handle this)
# However, if evaluate_results.py does NOT import them but expects them, we are safe.
# (Based on the prompt, evaluate_results.py likely has them defined or imported, 
# but defining them here doesn't hurt for dill context).

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/dmipy-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 2. Analyze Paths
    outer_pkl_path = None
    inner_pkl_path = None

    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_pkl_path = path
        elif 'standard_data_parent' in path and 'evaluate_results' in path:
            inner_pkl_path = path

    if not outer_pkl_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Load Outer Data
    try:
        with open(outer_pkl_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    outer_expected = outer_data.get('output', None)

    # 4. Execute Function
    print(f"Running evaluate_results from {outer_pkl_path}...")
    try:
        # Based on the provided code, evaluate_results returns an MSE float immediately,
        # it is NOT a factory function.
        # However, we must support the generic structure just in case logic changes.
        
        # Scenario A: Standard Execution
        result_obj = evaluate_results(*outer_args, **outer_kwargs)
        
        # Scenario B: Factory Pattern (Closure)
        # Check if we have inner data implying the result is a callable that needs to be called
        if inner_pkl_path and callable(result_obj):
            print(f"Detected Factory Pattern. Loading inner data from {inner_pkl_path}...")
            try:
                with open(inner_pkl_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Error loading inner pickle file: {e}")
                sys.exit(1)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            # The final result comes from the inner execution
            actual_result = result_obj(*inner_args, **inner_kwargs)
            expected_result = inner_data.get('output', None)
        else:
            # Simple function execution
            actual_result = result_obj
            expected_result = outer_expected

    except Exception as e:
        print(f"Error executing function: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    print("Verifying results...")
    try:
        is_correct, msg = recursive_check(expected_result, actual_result)
        
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Debug info
            print(f"Expected type: {type(expected_result)}")
            print(f"Actual type: {type(actual_result)}")
            if isinstance(expected_result, (int, float, np.number)):
                print(f"Expected: {expected_result}")
                print(f"Actual: {actual_result}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()