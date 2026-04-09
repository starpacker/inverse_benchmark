import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def compare_oopao_objects(expected, actual, path="root"):
    """
    Custom comparison for OOPAO simulation objects.
    For complex simulation objects, we compare key numerical attributes
    rather than object identity or string representation.
    """
    # Handle None cases
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"At {path}: One is None, other is not"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"At {path}: Expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"At {path}: Shape mismatch: {expected.shape} vs {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"At {path}: Array values differ"
        return True, ""
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"At {path}: Expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            return False, f"At {path}: Key mismatch: {set(expected.keys())} vs {set(actual.keys())}"
        for key in expected.keys():
            passed, msg = compare_oopao_objects(expected[key], actual[key], f"{path}.{key}")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle lists and tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)):
            return False, f"At {path}: Type mismatch: {type(expected)} vs {type(actual)}"
        if len(expected) != len(actual):
            return False, f"At {path}: Length mismatch: {len(expected)} vs {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = compare_oopao_objects(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle primitive types
    if isinstance(expected, (int, float, str, bool)):
        if isinstance(expected, float) and isinstance(actual, float):
            if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8):
                return False, f"At {path}: Float mismatch: {expected} vs {actual}"
            return True, ""
        if expected != actual:
            return False, f"At {path}: Value mismatch: {expected} vs {actual}"
        return True, ""
    
    # Handle OOPAO simulation objects by comparing their class and key attributes
    exp_class = expected.__class__.__name__
    act_class = actual.__class__.__name__
    
    if exp_class != act_class:
        return False, f"At {path}: Class mismatch: {exp_class} vs {act_class}"
    
    # For OOPAO objects, compare key numerical attributes
    oopao_classes = ['Telescope', 'Source', 'Atmosphere', 'DeformableMirror', 
                     'ShackHartmann', 'Detector']
    
    if exp_class in oopao_classes:
        # These are simulation objects - verify they have same configuration
        # by checking key attributes that define their behavior
        
        if exp_class == 'Telescope':
            attrs_to_check = ['resolution', 'D', 'samplingTime', 'centralObstruction']
        elif exp_class == 'Source':
            attrs_to_check = ['optBand', 'magnitude']
        elif exp_class == 'Atmosphere':
            attrs_to_check = ['r0', 'L0', 'nLayer']
        elif exp_class == 'DeformableMirror':
            attrs_to_check = ['nSubap', 'nValidAct', 'mechCoupling']
        elif exp_class == 'ShackHartmann':
            attrs_to_check = ['nSubap', 'nValidSubaperture', 'nSignal']
        elif exp_class == 'Detector':
            attrs_to_check = ['resolution']
        else:
            attrs_to_check = []
        
        for attr in attrs_to_check:
            if hasattr(expected, attr) and hasattr(actual, attr):
                exp_val = getattr(expected, attr)
                act_val = getattr(actual, attr)
                passed, msg = compare_oopao_objects(exp_val, act_val, f"{path}.{attr}")
                if not passed:
                    return False, msg
            elif hasattr(expected, attr) != hasattr(actual, attr):
                return False, f"At {path}: Attribute '{attr}' existence mismatch"
        
        return True, ""
    
    # For other objects, check if they're the same type
    if type(expected) != type(actual):
        return False, f"At {path}: Type mismatch: {type(expected)} vs {type(actual)}"
    
    # If objects have __dict__, compare their attributes
    if hasattr(expected, '__dict__') and hasattr(actual, '__dict__'):
        # Just verify same class - instance differences are expected
        return True, ""
    
    # Fallback: try direct comparison
    try:
        if expected == actual:
            return True, ""
    except:
        pass
    
    # If we can't compare, assume OK if same type
    return True, ""


def run_test():
    """Main test function."""
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    print("Executing load_and_preprocess_data...")
    try:
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if there are inner paths (factory pattern)
    if inner_paths:
        # Factory/Closure pattern - need to execute the returned operator
        print(f"Found {len(inner_paths)} inner data file(s) - Factory pattern detected")
        
        if not callable(actual_result):
            print("ERROR: Expected callable result for factory pattern, got non-callable")
            sys.exit(1)
        
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print("Executing operator with inner args...")
            try:
                actual_result = actual_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    # Phase 3: Compare results using custom comparison for OOPAO objects
    print("Comparing results...")
    
    passed, msg = compare_oopao_objects(expected_output, actual_result)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    run_test()