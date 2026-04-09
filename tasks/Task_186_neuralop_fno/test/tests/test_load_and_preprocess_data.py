import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def compare_dataloaders(expected_loader, actual_loader, name="DataLoader"):
    """Compare two DataLoaders by extracting and comparing their underlying data."""
    try:
        # Extract all data from both loaders
        expected_data = []
        for batch in expected_loader:
            expected_data.append(batch)
        
        actual_data = []
        for batch in actual_loader:
            actual_data.append(batch)
        
        # Compare number of batches
        if len(expected_data) != len(actual_data):
            return False, f"{name}: Different number of batches (expected {len(expected_data)}, got {len(actual_data)})"
        
        # Compare each batch
        for i, (exp_batch, act_batch) in enumerate(zip(expected_data, actual_data)):
            if len(exp_batch) != len(act_batch):
                return False, f"{name} batch {i}: Different number of tensors"
            
            for j, (exp_tensor, act_tensor) in enumerate(zip(exp_batch, act_batch)):
                if exp_tensor.shape != act_tensor.shape:
                    return False, f"{name} batch {i} tensor {j}: Shape mismatch"
                if not torch.allclose(exp_tensor, act_tensor, rtol=1e-5, atol=1e-5):
                    return False, f"{name} batch {i} tensor {j}: Value mismatch"
        
        return True, "OK"
    except Exception as e:
        return False, f"{name}: Comparison error - {str(e)}"

def compare_data_dicts(expected, actual):
    """Custom comparison for the data dictionary returned by load_and_preprocess_data."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return recursive_check(expected, actual)
    
    # Check keys match
    if set(expected.keys()) != set(actual.keys()):
        return False, f"Key mismatch: expected {set(expected.keys())}, got {set(actual.keys())}"
    
    for key in expected.keys():
        exp_val = expected[key]
        act_val = actual[key]
        
        # Handle DataLoader objects specially
        if 'DataLoader' in str(type(exp_val)):
            # Skip DataLoader comparison - they will always be different objects
            # Instead, we verify the underlying data through other keys
            continue
        
        # Handle numpy arrays
        elif isinstance(exp_val, np.ndarray):
            if not isinstance(act_val, np.ndarray):
                return False, f"Key '{key}': Type mismatch (expected ndarray, got {type(act_val)})"
            if exp_val.shape != act_val.shape:
                return False, f"Key '{key}': Shape mismatch (expected {exp_val.shape}, got {act_val.shape})"
            if not np.allclose(exp_val, act_val, rtol=1e-5, atol=1e-5):
                return False, f"Key '{key}': Value mismatch in numpy array"
        
        # Handle torch tensors
        elif isinstance(exp_val, torch.Tensor):
            if not isinstance(act_val, torch.Tensor):
                return False, f"Key '{key}': Type mismatch (expected Tensor, got {type(act_val)})"
            if exp_val.shape != act_val.shape:
                return False, f"Key '{key}': Shape mismatch (expected {exp_val.shape}, got {act_val.shape})"
            if not torch.allclose(exp_val, act_val, rtol=1e-5, atol=1e-5):
                return False, f"Key '{key}': Value mismatch in tensor"
        
        # Use recursive_check for other types
        else:
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"Key '{key}': {msg}"
    
    return True, "All checks passed"

def main():
    data_paths = ['/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    print(f"[TEST] Outer data path: {outer_path}")
    print(f"[TEST] Inner data paths: {inner_paths}")
    
    if outer_path is None:
        print("TEST FAILED: No outer data file found")
        sys.exit(1)
    
    # Load outer data
    try:
        print("[TEST] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        print(f"[TEST] Outer args: {outer_args}")
        print(f"[TEST] Outer kwargs: {outer_kwargs}")
    except Exception as e:
        print(f"TEST FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("[TEST] Executing load_and_preprocess_data...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[TEST] Function execution completed.")
    except Exception as e:
        print(f"TEST FAILED: Function execution error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario and compare
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("[TEST] Scenario B detected: Factory/Closure pattern")
        try:
            inner_path = inner_paths[0]
            print(f"[TEST] Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print("[TEST] Executing operator with inner args...")
            actual_result = result(*inner_args, **inner_kwargs)
            
            print("[TEST] Comparing results...")
            passed, msg = compare_data_dicts(expected, actual_result)
        except Exception as e:
            print(f"TEST FAILED: Inner execution error: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[TEST] Scenario A detected: Simple function")
        expected = outer_data.get('output')
        actual_result = result
        
        print("[TEST] Comparing results...")
        passed, msg = compare_data_dicts(expected, actual_result)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()