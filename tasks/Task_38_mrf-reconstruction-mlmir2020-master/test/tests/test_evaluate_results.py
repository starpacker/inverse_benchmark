import sys
import os
import dill
import torch
import numpy as np
import traceback
import argparse

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("CRITICAL: Could not import 'evaluate_results' from 'agent_evaluate_results'.")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing (for standalone testing robustness)
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray):
                return False, f"Expected numpy array, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
            if not np.allclose(expected, actual, rtol=1e-4, atol=1e-4):
                return False, "Values mismatch within tolerance"
            return True, "Arrays match"
        return expected == actual, f"Expected {expected}, got {actual}"

# Hardcoded data paths from instruction
DATA_PATHS = [
    '/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
]

def load_pickle_data(path):
    """Robust pickle loader with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    if os.path.getsize(path) == 0:
        raise EOFError(f"Data file is empty (0 bytes): {path}")

    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file {path}. Error: {str(e)}")

def run_test():
    print("DEBUG: Starting test_evaluate_results.py")
    
    # 1. Analyze Data Paths
    outer_path = None
    inner_path = None
    
    for p in DATA_PATHS:
        if 'standard_data_evaluate_results.pkl' in p:
            outer_path = p
        elif 'parent_function_evaluate_results' in p:
            inner_path = p
            
    print(f"DEBUG: Outer Path: {outer_path}")
    print(f"DEBUG: Inner Path: {inner_path}")

    if not outer_path:
        print("CRITICAL: No outer data file found for 'evaluate_results'.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        outer_data = load_pickle_data(outer_path)
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Determine Execution Strategy
    # Scenario A: evaluate_results is a standard function (returns results directly).
    # Scenario B: evaluate_results is a factory (returns a closure/operator).
    
    # Based on the provided reference code, evaluate_results returns `x_rec_denorm` (numpy array).
    # It is NOT a factory. So we strictly follow Scenario A logic.
    # If inner_path exists, it might be spurious or from a different test generation run, 
    # but for this specific function signature, we expect direct execution.
    
    func_args = outer_data.get('args', [])
    func_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    # 4. Device Compatibilty Patch
    # The recorded data might have tensors on GPU (cuda:x). 
    # We need to map them to the available device on the testing machine.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device: {device}")

    # Helper to move tensors to current device
    def move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        return obj

    # Apply device patch to args/kwargs
    func_args = move_to_device(func_args)
    func_kwargs = move_to_device(func_kwargs)
    
    # Specifically check the 'device' argument in evaluate_results signature:
    # (model, dataloader, dataset, dims, device, epochs) -> index 4
    if len(func_args) > 4:
        print(f"DEBUG: Overwriting argument at index 4 with current device {device}")
        func_args = list(func_args)
        func_args[4] = device
        func_args = tuple(func_args)

    # 5. Execute Function
    try:
        print("DEBUG: Executing evaluate_results...")
        actual_result = evaluate_results(*func_args, **func_kwargs)
        print("DEBUG: Execution successful.")
    except Exception as e:
        print(f"CRITICAL: Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Verification
    # If we have an inner path (Scenario B), it usually implies the output of the first call 
    # is a callable. But here we know it returns a numpy array.
    # We will ignore inner_path for execution if the result is not callable.
    
    if callable(actual_result) and inner_path:
        print("DEBUG: Result is callable and inner path exists. Switching to Scenario B (Factory).")
        try:
            inner_data = load_pickle_data(inner_path)
            inner_args = move_to_device(inner_data.get('args', []))
            inner_kwargs = move_to_device(inner_data.get('kwargs', {}))
            expected_output = inner_data.get('output')
            
            actual_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"CRITICAL: Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Compare Results
    passed, msg = recursive_check(expected_output, actual_result)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()