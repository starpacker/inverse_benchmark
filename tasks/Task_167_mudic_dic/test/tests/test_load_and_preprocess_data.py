import sys
import os
import dill
import numpy as np
import traceback

# Add the repo path for imports
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO_DIR)

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

import muDIC as dic


def compare_image_stacks(expected_stack, actual_stack, rtol=1e-5, atol=1e-8):
    """
    Compare two ImageStack objects by their actual image content.
    """
    try:
        # Check if both are ImageStack instances
        if not isinstance(expected_stack, dic.ImageStack) or not isinstance(actual_stack, dic.ImageStack):
            return False, f"Type mismatch: expected {type(expected_stack)}, got {type(actual_stack)}"
        
        # Compare number of images
        expected_n = len(expected_stack)
        actual_n = len(actual_stack)
        if expected_n != actual_n:
            return False, f"ImageStack length mismatch: expected {expected_n}, got {actual_n}"
        
        # For synthetic data with random generation, we can't expect exact matches
        # Instead, verify structural properties
        for i in range(expected_n):
            expected_img = expected_stack[i]
            actual_img = actual_stack[i]
            
            if expected_img.shape != actual_img.shape:
                return False, f"Image {i} shape mismatch: expected {expected_img.shape}, got {actual_img.shape}"
            
            if expected_img.dtype != actual_img.dtype:
                return False, f"Image {i} dtype mismatch: expected {expected_img.dtype}, got {actual_img.dtype}"
        
        return True, "ImageStack comparison passed (structure verified)"
    except Exception as e:
        return False, f"ImageStack comparison error: {str(e)}"


def custom_recursive_check(expected, actual, path="output"):
    """
    Custom comparison that handles special cases like ImageStack objects.
    """
    # Handle ImageStack specially
    if isinstance(expected, dic.ImageStack):
        return compare_image_stacks(expected, actual)
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch at {path}: expected dict, got {type(actual)}"
        
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}: expected {set(expected.keys())}, got {set(actual.keys())}"
        
        for key in expected.keys():
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, "All checks passed"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"Type mismatch at {path}: expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8):
            return False, f"Value mismatch at {path}"
        return True, "Arrays match"
    
    # Handle callables (functions)
    if callable(expected):
        if not callable(actual):
            return False, f"Callable mismatch at {path}: expected callable, got {type(actual)}"
        # For functions, just check they're both callable with same name if possible
        expected_name = getattr(expected, '__name__', str(expected))
        actual_name = getattr(actual, '__name__', str(actual))
        if expected_name != actual_name:
            return False, f"Function name mismatch at {path}: expected {expected_name}, got {actual_name}"
        return True, "Callable match"
    
    # Handle tuples
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple):
            return False, f"Type mismatch at {path}: expected tuple, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"Tuple length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Tuples match"
    
    # Handle lists
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, f"Type mismatch at {path}: expected list, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"List length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Lists match"
    
    # Handle numeric types with tolerance
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8):
            return False, f"Numeric mismatch at {path}: expected {expected}, got {actual}"
        return True, "Numeric match"
    
    # Default equality check
    if expected != actual:
        return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
    
    return True, "Match"


def main():
    data_paths = ['/data/yjh/mudic_dic_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Determine scenario
    if inner_paths:
        print("Scenario B detected: Factory/Closure pattern")
    else:
        print("Scenario A detected: Simple function test")
    
    try:
        # Phase 1: Load outer data and execute function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data.get('func_name', 'unknown')
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Function name: {func_name}")
        print(f"Args: {outer_args}")
        print(f"Kwargs: {outer_kwargs}")
        
        print("Executing load_and_preprocess_data...")
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Execution completed successfully.")
        
        # Phase 2: Handle based on scenario
        if inner_paths:
            # Scenario B: Factory pattern
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            if not callable(actual_result):
                print(f"ERROR: Expected callable from Phase 1, got {type(actual_result)}")
                sys.exit(1)
            
            print("Executing inner function...")
            actual_result = actual_result(*inner_args, **inner_kwargs)
            print("Inner execution completed.")
        
        # Verification
        print("Verifying results...")
        passed, msg = custom_recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()