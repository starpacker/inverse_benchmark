import sys
import os
import dill
import numpy as np
import traceback

# Add the path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_datacubes(expected, actual, rtol=1e-5, atol=1e-8):
    """Compare two DataCube objects by their data and calibration."""
    try:
        # Compare the underlying data arrays
        if not np.allclose(expected.data, actual.data, rtol=rtol, atol=atol, equal_nan=True):
            return False, "DataCube data arrays do not match"
        
        # Compare calibrations if they exist
        if hasattr(expected, 'calibration') and hasattr(actual, 'calibration'):
            exp_cal = expected.calibration
            act_cal = actual.calibration
            
            # Compare R pixel size
            if hasattr(exp_cal, 'get_R_pixel_size') and hasattr(act_cal, 'get_R_pixel_size'):
                exp_r = exp_cal.get_R_pixel_size()
                act_r = act_cal.get_R_pixel_size()
                if exp_r is not None and act_r is not None:
                    if not np.isclose(exp_r, act_r, rtol=rtol, atol=atol):
                        return False, f"R pixel size mismatch: {exp_r} vs {act_r}"
            
            # Compare Q pixel size
            if hasattr(exp_cal, 'get_Q_pixel_size') and hasattr(act_cal, 'get_Q_pixel_size'):
                exp_q = exp_cal.get_Q_pixel_size()
                act_q = act_cal.get_Q_pixel_size()
                if exp_q is not None and act_q is not None:
                    if not np.isclose(exp_q, act_q, rtol=rtol, atol=atol):
                        return False, f"Q pixel size mismatch: {exp_q} vs {act_q}"
        
        return True, "DataCubes match"
    except Exception as e:
        return False, f"Error comparing DataCubes: {str(e)}"


def custom_recursive_check(expected, actual, path="output", rtol=1e-5, atol=1e-8):
    """Custom recursive check that handles DataCube objects."""
    try:
        # Check for DataCube type
        if hasattr(expected, 'data') and hasattr(actual, 'data') and \
           expected.__class__.__name__ == 'DataCube' and actual.__class__.__name__ == 'DataCube':
            passed, msg = compare_datacubes(expected, actual, rtol, atol)
            if not passed:
                return False, f"Value mismatch at {path}: {msg}"
            return True, "Match"
        
        # Handle dictionaries
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False, f"Key mismatch at {path}: expected {set(expected.keys())}, got {set(actual.keys())}"
            for key in expected.keys():
                passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']", rtol, atol)
                if not passed:
                    return False, msg
            return True, "Match"
        
        # Handle numpy arrays
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            if expected.shape != actual.shape:
                return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
            if expected.dtype != actual.dtype:
                # Allow dtype differences if values are close
                pass
            if np.iscomplexobj(expected) or np.iscomplexobj(actual):
                if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                    return False, f"Complex array value mismatch at {path}"
            else:
                if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                    return False, f"Array value mismatch at {path}"
            return True, "Match"
        
        # Handle lists and tuples
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(expected) != len(actual):
                return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
            for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                passed, msg = custom_recursive_check(exp_item, act_item, f"{path}[{i}]", rtol, atol)
                if not passed:
                    return False, msg
            return True, "Match"
        
        # Handle numeric scalars
        if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            if not np.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                return False, f"Numeric mismatch at {path}: expected {expected}, got {actual}"
            return True, "Match"
        
        # Handle None
        if expected is None and actual is None:
            return True, "Match"
        
        # Handle strings
        if isinstance(expected, str) and isinstance(actual, str):
            if expected != actual:
                return False, f"String mismatch at {path}: expected '{expected}', got '{actual}'"
            return True, "Match"
        
        # Fallback: try direct comparison
        try:
            if expected == actual:
                return True, "Match"
        except:
            pass
        
        # If types are the same, assume match for complex objects
        if type(expected).__name__ == type(actual).__name__:
            return True, "Match (same type)"
        
        return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
        
    except Exception as e:
        return False, f"Error during comparison at {path}: {str(e)}"


def main():
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    print("Executing load_and_preprocess_data...")
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Factory pattern with {len(inner_paths)} inner test(s)")
        
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            # The result should be callable
            if not callable(result):
                print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
                sys.exit(1)
            
            print("Executing inner function...")
            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            passed, msg = custom_recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function test
        print("\nScenario A detected: Simple function test")
        expected = outer_data.get('output')
        
        passed, msg = custom_recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()