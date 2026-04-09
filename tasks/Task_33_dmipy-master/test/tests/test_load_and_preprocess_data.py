import sys
import os
import dill
import numpy as np
import traceback
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Helpers required for dill loading if they were captured in the closure (though this specific function is flat)
def unitsphere2cart_1d(theta, phi):
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def fibonacci_sphere(samples=60):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

# Inject helpers into global namespace so dill can find them if needed during deserialization
globals()['unitsphere2cart_1d'] = unitsphere2cart_1d
globals()['fibonacci_sphere'] = fibonacci_sphere

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/dmipy-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    outer_path = None
    inner_path = None

    for p in data_paths:
        if 'parent_function' in p:
            inner_path = p
        elif 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_load_and_preprocess_data.pkl not found.")
        sys.exit(1)

    # 2. Load Outer Data
    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer pickle: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. Execution Strategy
    actual_result = None

    try:
        print("Executing target function 'load_and_preprocess_data'...")
        # Scenario A: Direct Execution (No factory pattern detected based on provided paths)
        # However, we check if the result is callable just in case.
        temp_result = load_and_preprocess_data(*outer_args, **outer_kwargs)

        if inner_path:
            # Scenario B: Factory Pattern
            print(f"Factory pattern detected. Loading inner data: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            if not callable(temp_result):
                print("Error: Expected a callable from outer function for factory pattern, got something else.")
                sys.exit(1)
                
            actual_result = temp_result(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple Function
            print("Detected Direct Execution Pattern.")
            actual_result = temp_result

    except Exception as e:
        print("Execution failed.")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    
    # Custom verification logic for stochastic output
    # The output is (bvalues, gradient_directions, signal_noisy, gt_params)
    # signal_noisy (index 2) contains random noise and will fail strict equality checks.
    
    if isinstance(expected_output, tuple) and len(expected_output) == 4:
        # Check deterministic components
        deterministic_indices = [0, 1, 3] # bvalues, gradients, gt_params
        for idx in deterministic_indices:
            passed, msg = recursive_check(expected_output[idx], actual_result[idx])
            if not passed:
                print(f"FAILED: Deterministic mismatch at index {idx}: {msg}")
                sys.exit(1)
        
        # Check stochastic component (signal_noisy) structurally
        stochastic_idx = 2
        exp_sig = expected_output[stochastic_idx]
        act_sig = actual_result[stochastic_idx]
        
        if type(exp_sig) != type(act_sig):
            print(f"FAILED: Type mismatch at index 2. Expected {type(exp_sig)}, got {type(act_sig)}")
            sys.exit(1)
            
        if hasattr(exp_sig, 'shape') and hasattr(act_sig, 'shape'):
            if exp_sig.shape != act_sig.shape:
                print(f"FAILED: Shape mismatch at index 2. Expected {exp_sig.shape}, got {act_sig.shape}")
                sys.exit(1)
        
        print("Stochastic component (index 2) structure verified (skipping value check due to random noise).")
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Fallback for unexpected output structure
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()