import sys
import os
import dill
import numpy as np
import traceback

# Set random seed for reproducibility
np.random.seed(42)

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def compare_sunpy_maps(expected, actual):
    """Compare two SunPy Map objects by their data arrays."""
    try:
        import sunpy.map
        if isinstance(expected, sunpy.map.GenericMap) and isinstance(actual, sunpy.map.GenericMap):
            # Compare the underlying data arrays
            return np.allclose(expected.data, actual.data, rtol=1e-5, atol=1e-8)
        return None  # Not SunPy maps
    except ImportError:
        return None

def custom_recursive_check(expected, actual, path="root"):
    """Custom comparison that handles SunPy Map objects."""
    try:
        import sunpy.map
        
        # Handle SunPy Map objects
        if isinstance(expected, sunpy.map.GenericMap) and isinstance(actual, sunpy.map.GenericMap):
            if np.allclose(expected.data, actual.data, rtol=1e-5, atol=1e-8):
                return True, "SunPy Map data matches"
            else:
                return False, f"SunPy Map data mismatch at {path}"
        
        # Handle tuples
        if isinstance(expected, tuple) and isinstance(actual, tuple):
            if len(expected) != len(actual):
                return False, f"Tuple length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
            for i, (e, a) in enumerate(zip(expected, actual)):
                passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
                if not passed:
                    return False, msg
            return True, "All tuple elements match"
        
        # Handle lists
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False, f"List length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
            for i, (e, a) in enumerate(zip(expected, actual)):
                passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
                if not passed:
                    return False, msg
            return True, "All list elements match"
        
        # Handle numpy arrays
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            if np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
                return True, "Arrays match"
            else:
                return False, f"Array mismatch at {path}"
        
        # Handle dicts
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False, f"Dict keys mismatch at {path}"
            for k in expected:
                passed, msg = custom_recursive_check(expected[k], actual[k], f"{path}[{k}]")
                if not passed:
                    return False, msg
            return True, "All dict elements match"
        
        # Fall back to recursive_check for other types
        return recursive_check(expected, actual)
        
    except ImportError:
        # If sunpy not available, use default check
        return recursive_check(expected, actual)

def main():
    data_paths = ['/data/yjh/pfsspy_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
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
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Scenario A: Simple function (no inner paths)
    if not inner_paths:
        try:
            # Reset seed before calling function to match data generation
            np.random.seed(42)
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("Function executed successfully")
        except Exception as e:
            print(f"ERROR executing function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results using custom checker for SunPy maps
        try:
            passed, msg = custom_recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern
    else:
        try:
            np.random.seed(42)
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("Operator created successfully")
        except Exception as e:
            print(f"ERROR creating operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print("ERROR: Returned operator is not callable")
            sys.exit(1)
        
        # Load inner data and execute
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from {inner_path}")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Inner function executed successfully")
        except Exception as e:
            print(f"ERROR executing inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = custom_recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()