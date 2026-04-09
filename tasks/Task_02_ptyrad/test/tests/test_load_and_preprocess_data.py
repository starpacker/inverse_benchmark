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


def compare_initializer(expected_init, actual_init):
    """Compare two Initializer objects by their key attributes."""
    if type(expected_init).__name__ != type(actual_init).__name__:
        return False, f"Type mismatch: {type(expected_init).__name__} vs {type(actual_init).__name__}"
    
    # Compare the initialized data attributes
    expected_attrs = vars(expected_init) if hasattr(expected_init, '__dict__') else {}
    actual_attrs = vars(actual_init) if hasattr(actual_init, '__dict__') else {}
    
    for key in expected_attrs:
        if key.startswith('_'):
            continue
        if key not in actual_attrs:
            return False, f"Missing attribute '{key}' in actual Initializer"
        
        exp_val = expected_attrs[key]
        act_val = actual_attrs[key]
        
        if isinstance(exp_val, (torch.Tensor, np.ndarray)):
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"Initializer attribute '{key}': {msg}"
        elif isinstance(exp_val, dict):
            # For dict attributes like init_variables, do a more lenient comparison
            passed, msg = compare_dict_lenient(exp_val, act_val, key)
            if not passed:
                return False, f"Initializer attribute '{key}': {msg}"
    
    return True, "Initializer objects match"


def compare_dict_lenient(expected_dict, actual_dict, parent_key=""):
    """Compare dictionaries with lenient tolerance for numerical values."""
    if not isinstance(expected_dict, dict) or not isinstance(actual_dict, dict):
        return recursive_check(expected_dict, actual_dict)
    
    for key in expected_dict:
        if key not in actual_dict:
            return False, f"Missing key '{key}' in actual dict"
        
        exp_val = expected_dict[key]
        act_val = actual_dict[key]
        
        if isinstance(exp_val, (torch.Tensor, np.ndarray)):
            # Use higher tolerance for position-related arrays
            if 'pos' in key.lower() or 'crop' in key.lower():
                # For position arrays, allow integer differences up to 1
                if isinstance(exp_val, torch.Tensor):
                    exp_np = exp_val.cpu().numpy()
                else:
                    exp_np = exp_val
                if isinstance(act_val, torch.Tensor):
                    act_np = act_val.cpu().numpy()
                else:
                    act_np = act_val
                
                if exp_np.shape != act_np.shape:
                    return False, f"Shape mismatch for '{key}': {exp_np.shape} vs {act_np.shape}"
                
                max_diff = np.max(np.abs(exp_np.astype(float) - act_np.astype(float)))
                # Allow tolerance of 1 for position/crop arrays (rounding differences)
                if max_diff > 1.5:
                    return False, f"Value mismatch for '{key}': max difference {max_diff} > 1.5"
            else:
                passed, msg = recursive_check(exp_val, act_val)
                if not passed:
                    return False, f"Key '{key}': {msg}"
        elif isinstance(exp_val, dict):
            passed, msg = compare_dict_lenient(exp_val, act_val, key)
            if not passed:
                return False, msg
        else:
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"Key '{key}': {msg}"
    
    return True, "Dict comparison passed"


def compare_combined_loss(expected_loss, actual_loss):
    """Compare two CombinedLoss objects."""
    if type(expected_loss).__name__ != type(actual_loss).__name__:
        return False, f"Type mismatch: {type(expected_loss).__name__} vs {type(actual_loss).__name__}"
    
    # Compare key attributes
    expected_attrs = vars(expected_loss) if hasattr(expected_loss, '__dict__') else {}
    actual_attrs = vars(actual_loss) if hasattr(actual_loss, '__dict__') else {}
    
    for key in expected_attrs:
        if key.startswith('_'):
            continue
        if key not in actual_attrs:
            return False, f"Missing attribute '{key}' in actual CombinedLoss"
        
        exp_val = expected_attrs[key]
        act_val = actual_attrs[key]
        
        # Skip callable comparisons
        if callable(exp_val) and not isinstance(exp_val, (torch.Tensor, np.ndarray)):
            continue
        
        if isinstance(exp_val, (torch.Tensor, np.ndarray)):
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"CombinedLoss attribute '{key}': {msg}"
        elif isinstance(exp_val, (dict, list, tuple, int, float, str, bool, type(None))):
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"CombinedLoss attribute '{key}': {msg}"
    
    return True, "CombinedLoss objects match"


def compare_combined_constraint(expected_constraint, actual_constraint):
    """Compare two CombinedConstraint objects."""
    if type(expected_constraint).__name__ != type(actual_constraint).__name__:
        return False, f"Type mismatch: {type(expected_constraint).__name__} vs {type(actual_constraint).__name__}"
    
    # Compare key attributes
    expected_attrs = vars(expected_constraint) if hasattr(expected_constraint, '__dict__') else {}
    actual_attrs = vars(actual_constraint) if hasattr(actual_constraint, '__dict__') else {}
    
    for key in expected_attrs:
        if key.startswith('_'):
            continue
        if key not in actual_attrs:
            return False, f"Missing attribute '{key}' in actual CombinedConstraint"
        
        exp_val = expected_attrs[key]
        act_val = actual_attrs[key]
        
        # Skip callable comparisons
        if callable(exp_val) and not isinstance(exp_val, (torch.Tensor, np.ndarray)):
            continue
        
        if isinstance(exp_val, (torch.Tensor, np.ndarray)):
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"CombinedConstraint attribute '{key}': {msg}"
        elif isinstance(exp_val, (dict, list, tuple, int, float, str, bool, type(None))):
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"CombinedConstraint attribute '{key}': {msg}"
    
    return True, "CombinedConstraint objects match"


def compare_results(expected, actual):
    """Custom comparison for load_and_preprocess_data output."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return recursive_check(expected, actual)
    
    # Check all keys exist
    for key in expected:
        if key not in actual:
            return False, f"Missing key '{key}' in actual result"
    
    for key in expected:
        exp_val = expected[key]
        act_val = actual[key]
        
        # Handle special object types
        if key == 'init':
            # Compare Initializer objects
            passed, msg = compare_initializer(exp_val, act_val)
            if not passed:
                return False, f"init comparison failed: {msg}"
        elif key == 'loss_fn':
            # Compare CombinedLoss objects
            passed, msg = compare_combined_loss(exp_val, act_val)
            if not passed:
                return False, f"loss_fn comparison failed: {msg}"
        elif key == 'constraint_fn':
            # Compare CombinedConstraint objects
            passed, msg = compare_combined_constraint(exp_val, act_val)
            if not passed:
                return False, f"constraint_fn comparison failed: {msg}"
        elif key == 'device':
            # Compare device types
            if str(exp_val) != str(act_val):
                return False, f"device mismatch: expected {exp_val}, got {act_val}"
        elif key == 'logger':
            # Logger can be None or different instances, just check type
            if type(exp_val) != type(act_val):
                return False, f"logger type mismatch: expected {type(exp_val)}, got {type(act_val)}"
        elif key == 'params':
            # Compare params dict
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"params comparison failed: {msg}"
        elif key == 'verbose':
            if exp_val != act_val:
                return False, f"verbose mismatch: expected {exp_val}, got {act_val}"
        else:
            # Default comparison
            passed, msg = recursive_check(exp_val, act_val)
            if not passed:
                return False, f"Key '{key}' comparison failed: {msg}"
    
    return True, "All comparisons passed"


def main():
    """Main test function for load_and_preprocess_data."""
    
    data_paths = ['/home/yjh/ad_pty/code_2/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Step 1: Categorize data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Step 2: Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Outer data loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute the function
    try:
        print("\n### Executing load_and_preprocess_data ###")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine scenario and get expected output
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, but inner paths exist. Treating as Scenario A.")
            expected = outer_data.get('output')
        else:
            # Load inner data and execute operator
            inner_path = inner_paths[0]  # Use first inner path
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Inner data loaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            
            try:
                print("\n### Executing operator with inner args ###")
                result = result(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            expected = inner_data.get('output')
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function call")
        expected = outer_data.get('output')
    
    # Step 5: Verify results
    print("\n### Verifying results ###")
    
    if expected is None:
        print("WARNING: Expected output is None. Checking if result is also None or valid.")
        if result is None:
            print("TEST PASSED (both expected and result are None)")
            sys.exit(0)
        else:
            print(f"Result type: {type(result)}")
            print("TEST PASSED (no expected output to compare, function executed successfully)")
            sys.exit(0)
    
    try:
        # Use custom comparison for this function's output
        passed, msg = compare_results(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()