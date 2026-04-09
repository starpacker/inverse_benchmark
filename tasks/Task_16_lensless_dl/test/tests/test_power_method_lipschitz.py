import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_power_method_lipschitz import power_method_lipschitz
from verification_utils import recursive_check


def check_with_tolerance(expected, actual, rtol=1e-4, atol=1e-6):
    """
    Check if expected and actual values are close within tolerance.
    Returns (passed, message) tuple.
    """
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(actual, (int, float, np.integer, np.floating)):
            if np.isclose(expected, actual, rtol=rtol, atol=atol):
                return True, "Values match within tolerance"
            else:
                return False, f"Value mismatch: expected {expected}, got {actual}, diff={abs(expected-actual)}"
        else:
            return False, f"Type mismatch: expected numeric, got {type(actual)}"
    elif isinstance(expected, np.ndarray):
        if isinstance(actual, np.ndarray):
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
            if np.allclose(expected, actual, rtol=rtol, atol=atol):
                return True, "Arrays match within tolerance"
            else:
                max_diff = np.max(np.abs(expected - actual))
                return False, f"Array mismatch: max difference = {max_diff}"
        else:
            return False, f"Type mismatch: expected ndarray, got {type(actual)}"
    else:
        # Fall back to recursive_check for other types
        return recursive_check(expected, actual)


def main():
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_power_method_lipschitz.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_power_method_lipschitz.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_power_method_lipschitz.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Phase 2: Execute the function
    try:
        # For power_method_lipschitz, it returns a Lipschitz constant (scalar)
        # Not a callable operator
        result = power_method_lipschitz(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
        print(f"Result type: {type(result)}")
        print(f"Result value: {result}")
    except Exception as e:
        print(f"ERROR executing power_method_lipschitz: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Check if there are inner paths (factory pattern)
    if inner_paths:
        # This would be the factory pattern case
        print(f"Found {len(inner_paths)} inner data file(s)")
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                # Execute the operator with inner args
                inner_result = result(*inner_args, **inner_kwargs)
                
                # Compare with tolerance
                passed, msg = check_with_tolerance(inner_expected, inner_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    print(f"Expected: {inner_expected}")
                    print(f"Actual: {inner_result}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly with expected output
        print(f"No inner data files - using simple comparison")
        print(f"Expected output: {expected_output}")
        print(f"Actual result: {result}")
        
        # Use tolerance-based comparison for numerical results
        passed, msg = check_with_tolerance(expected_output, result, rtol=1e-3, atol=1e-5)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_output}")
            print(f"Actual: {result}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()