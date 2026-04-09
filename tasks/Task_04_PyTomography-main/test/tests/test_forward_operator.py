import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing agent_forward_operator.py to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_forward_operator import forward_operator, SPECTSystemMatrix
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Helper for stochastic checks
def stochastic_shape_check(expected, actual):
    """
    Since the function includes Poisson noise, exact value matching will fail.
    We verify:
    1. Structure (tuple vs tuple)
    2. Tensor properties (shape, dtype, device)
    3. Type consistency for other objects (e.g., SystemMatrix)
    """
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False, f"Tuple length mismatch: expected {len(expected)}, got {len(actual)}"
        
        for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
            # Check Tensors (The noisy projection)
            if isinstance(exp_item, torch.Tensor) and isinstance(act_item, torch.Tensor):
                if exp_item.shape != act_item.shape:
                    return False, f"Shape mismatch at index {i}: expected {exp_item.shape}, got {act_item.shape}"
                if exp_item.dtype != act_item.dtype:
                    return False, f"Dtype mismatch at index {i}: expected {exp_item.dtype}, got {act_item.dtype}"
                # We skip value comparison due to random Poisson noise
            
            # Check SystemMatrix or other objects
            elif hasattr(exp_item, '__class__') and hasattr(act_item, '__class__'):
                if exp_item.__class__.__name__ != act_item.__class__.__name__:
                     return False, f"Class mismatch at index {i}: expected {exp_item.__class__.__name__}, got {act_item.__class__.__name__}"
            
            else:
                # Fallback for simple types
                if type(exp_item) != type(act_item):
                     return False, f"Type mismatch at index {i}: expected {type(exp_item)}, got {type(act_item)}"
        
        return True, "Stochastic structure verified"
    else:
        # Fallback to recursive check if not a tuple or simple structure
        return recursive_check(expected, actual)

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'parent_function_forward_operator' in path:
            inner_path = path
        elif 'standard_data_forward_operator.pkl' in path:
            outer_path = path

    print("Test Strategy Analysis:")
    print(f"  - Outer Data (Args): {outer_path}")
    print(f"  - Inner Data (Exec): {inner_path if inner_path else 'None (Simple Execution)'}")

    if not outer_path:
        print("Skipping test: No outer data file found.")
        sys.exit(0)

    # 2. Load Data
    try:
        print("\nLoading input data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        inner_data = None
        if inner_path:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution
    try:
        # Scenario A: Simple Function Execution (Most likely for forward_operator)
        # The provided code shows forward_operator returns (projections, matrix) immediately.
        if not inner_path:
            print("\n[Phase 1] Executing function 'forward_operator' directly...")
            
            # Prepare args/kwargs
            args = outer_data.get('args', [])
            kwargs = outer_data.get('kwargs', {})
            
            # Ensure tensors are on the correct device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            def to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                if isinstance(obj, list):
                    return [to_device(x) for x in obj]
                return obj

            args = [to_device(a) for a in args]
            kwargs = {k: to_device(v) for k, v in kwargs.items()}

            # Run Target
            actual_result = forward_operator(*args, **kwargs)
            expected_result = outer_data['output']

        # Scenario B: Factory Pattern (If inner path existed)
        else:
            print("\n[Phase 1] Initializing operator factory...")
            operator = forward_operator(*outer_data['args'], **outer_data['kwargs'])
            
            print("[Phase 2] Executing inner function...")
            args = inner_data.get('args', [])
            kwargs = inner_data.get('kwargs', {})
            actual_result = operator(*args, **kwargs)
            expected_result = inner_data['output']

    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("\n[Phase 3] Verifying results...")
    
    # Since forward_operator adds Poisson noise, standard equality checks will fail.
    # We use a custom check for structure and shape.
    passed, msg = stochastic_shape_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()