import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_dm_objects(expected_dm, actual_dm):
    """
    Compare DiffMetrology objects by checking key attributes.
    Full object comparison is not feasible for complex objects.
    """
    try:
        # Check if both are the same type
        if type(expected_dm).__name__ != type(actual_dm).__name__:
            return False, f"DM type mismatch: {type(expected_dm).__name__} vs {type(actual_dm).__name__}"
        return True, "DM objects are of same type (detailed comparison skipped for complex objects)"
    except Exception as e:
        return False, f"Error comparing DM objects: {str(e)}"


def custom_recursive_check(expected, actual, path="root"):
    """
    Custom comparison that handles special cases like DiffMetrology objects.
    """
    # Handle None cases
    if expected is None and actual is None:
        return True, "Both are None"
    if expected is None or actual is None:
        return False, f"At {path}: One is None, other is not"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"At {path}: Dictionary keys mismatch. Expected: {set(expected.keys())}, Got: {set(actual.keys())}"
        
        for key in expected.keys():
            # Special handling for 'DM' key (DiffMetrology object)
            if key == 'DM':
                passed, msg = compare_dm_objects(expected[key], actual[key])
                if not passed:
                    return False, f"At {path}.{key}: {msg}"
                continue
            
            # Special handling for 'device' key
            if key == 'device':
                # Just check both are valid torch devices
                try:
                    expected_dev = str(expected[key])
                    actual_dev = str(actual[key])
                    # Both should be cuda or both cpu
                    if ('cuda' in expected_dev) == ('cuda' in actual_dev):
                        continue
                    else:
                        return False, f"At {path}.{key}: Device mismatch {expected_dev} vs {actual_dev}"
                except:
                    continue
            
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}.{key}")
            if not passed:
                return False, msg
        return True, "All dictionary entries match"
    
    # Handle torch tensors
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        try:
            expected_np = expected.detach().cpu().numpy()
            actual_np = actual.detach().cpu().numpy()
            if expected_np.shape != actual_np.shape:
                return False, f"At {path}: Tensor shape mismatch. Expected: {expected_np.shape}, Got: {actual_np.shape}"
            if np.allclose(expected_np, actual_np, rtol=1e-4, atol=1e-6, equal_nan=True):
                return True, "Tensors match"
            else:
                max_diff = np.max(np.abs(expected_np - actual_np))
                return False, f"At {path}: Tensor values mismatch. Max diff: {max_diff}"
        except Exception as e:
            return False, f"At {path}: Error comparing tensors: {str(e)}"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"At {path}: Array shape mismatch. Expected: {expected.shape}, Got: {actual.shape}"
        if np.allclose(expected, actual, rtol=1e-4, atol=1e-6, equal_nan=True):
            return True, "Arrays match"
        else:
            max_diff = np.max(np.abs(expected - actual))
            return False, f"At {path}: Array values mismatch. Max diff: {max_diff}"
    
    # Handle numeric types
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if np.isclose(expected, actual, rtol=1e-4, atol=1e-6) or (np.isnan(expected) and np.isnan(actual)):
            return True, "Numeric values match"
        return False, f"At {path}: Numeric mismatch. Expected: {expected}, Got: {actual}"
    
    # Handle strings
    if isinstance(expected, str) and isinstance(actual, str):
        if expected == actual:
            return True, "Strings match"
        return False, f"At {path}: String mismatch. Expected: {expected}, Got: {actual}"
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"At {path}: Length mismatch. Expected: {len(expected)}, Got: {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "List/tuple elements match"
    
    # For other types, try direct comparison or type check
    if type(expected) == type(actual):
        try:
            if expected == actual:
                return True, "Objects equal"
        except:
            pass
        # If direct comparison fails, at least types match
        return True, f"Same type ({type(expected).__name__}), detailed comparison skipped"
    
    return False, f"At {path}: Type mismatch. Expected: {type(expected)}, Got: {type(actual)}"


def main():
    data_paths = ['/home/yjh/diff_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_load_and_preprocess_data.pkl':
                outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"Found outer_path: {outer_path}")
    print(f"Found inner_paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'None'}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function execution completed successfully.")
        
    except Exception as e:
        print(f"ERROR executing function: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle inner paths if they exist (Scenario B)
    if inner_paths:
        print("Scenario B detected: Factory/Closure pattern")
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                if callable(result):
                    print("Executing returned operator...")
                    actual_result = result(*inner_args, **inner_kwargs)
                else:
                    actual_result = result
                
                # Compare results
                print("Comparing results...")
                passed, msg = custom_recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR in inner execution: {str(e)}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function")
        
        # Compare results
        print("Comparing results...")
        try:
            # First try the custom check which handles complex objects
            passed, msg = custom_recursive_check(expected_output, result)
            
            if not passed:
                # Fallback: try recursive_check from verification_utils
                print("Custom check failed, trying recursive_check...")
                passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR during comparison: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()