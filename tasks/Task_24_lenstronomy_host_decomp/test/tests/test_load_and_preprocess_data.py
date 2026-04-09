import sys
import os
import dill
import numpy as np
import traceback

# Add the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_lenstronomy_objects(expected, actual, path="output"):
    """
    Custom comparison for lenstronomy objects and complex nested structures.
    Returns (passed, message).
    """
    # Handle None cases
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"At {path}: one is None, other is not"
    
    # Get type names
    expected_type = type(expected).__name__
    actual_type = type(actual).__name__
    
    # Check type match
    if expected_type != actual_type:
        return False, f"At {path}: type mismatch - expected {expected_type}, got {actual_type}"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"At {path}: array shape mismatch - expected {expected.shape}, got {actual.shape}"
        # For image data with noise, we check if they're reasonably close or same structure
        # Since noise is random, we just check shapes and that values are finite
        if not np.all(np.isfinite(actual)):
            return False, f"At {path}: array contains non-finite values"
        return True, ""
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            missing = set(expected.keys()) - set(actual.keys())
            extra = set(actual.keys()) - set(expected.keys())
            return False, f"At {path}: dict keys mismatch. Missing: {missing}, Extra: {extra}"
        for key in expected.keys():
            passed, msg = compare_lenstronomy_objects(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle lists
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False, f"At {path}: list length mismatch - expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = compare_lenstronomy_objects(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle tuples
    if isinstance(expected, tuple):
        if len(expected) != len(actual):
            return False, f"At {path}: tuple length mismatch - expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = compare_lenstronomy_objects(e, a, f"{path}({i})")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle primitive types
    if isinstance(expected, (int, float, str, bool)):
        if isinstance(expected, float):
            if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
                return False, f"At {path}: float mismatch - expected {expected}, got {actual}"
            return True, ""
        if expected != actual:
            return False, f"At {path}: value mismatch - expected {expected}, got {actual}"
        return True, ""
    
    # Handle lenstronomy and other complex objects by checking their class name and key attributes
    lenstronomy_classes = ['ImageData', 'PSF', 'LightModel', 'PointSource', 'ImageModel']
    
    if expected_type in lenstronomy_classes:
        # For lenstronomy objects, just verify they are the same type
        # The actual content depends on random noise, so we can't compare exactly
        return True, ""
    
    # For other objects, try to compare their __dict__ if available
    if hasattr(expected, '__dict__') and hasattr(actual, '__dict__'):
        return compare_lenstronomy_objects(expected.__dict__, actual.__dict__, f"{path}.__dict__")
    
    # Fallback: just check types match (already done above)
    return True, ""


def validate_output_structure(result):
    """
    Validate that the output has the expected structure and types.
    """
    required_keys = [
        'data_class', 'psf_class', 'lightModel', 'pointSource',
        'image_sim', 'kwargs_data', 'kwargs_psf', 'kwargs_numerics',
        'kwargs_host', 'kwargs_ps', 'light_model_list', 'point_source_list'
    ]
    
    if not isinstance(result, dict):
        return False, f"Result should be dict, got {type(result).__name__}"
    
    missing_keys = set(required_keys) - set(result.keys())
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    
    # Validate specific types
    if not isinstance(result['image_sim'], np.ndarray):
        return False, f"image_sim should be ndarray, got {type(result['image_sim']).__name__}"
    
    if not isinstance(result['kwargs_data'], dict):
        return False, f"kwargs_data should be dict, got {type(result['kwargs_data']).__name__}"
    
    if not isinstance(result['kwargs_host'], list):
        return False, f"kwargs_host should be list, got {type(result['kwargs_host']).__name__}"
    
    if not isinstance(result['light_model_list'], list):
        return False, f"light_model_list should be list, got {type(result['light_model_list']).__name__}"
    
    return True, ""


def main():
    data_paths = ['/home/yjh/lenstronomy_host_decomp_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data with outer args/kwargs...")
        
        # Set random seed for reproducibility if testing structure
        # Note: The function uses np.random internally, so results will differ
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
        print(f"Function returned type: {type(result).__name__}")
        
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle inner paths if they exist (Scenario B)
    if inner_paths:
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        # For closure pattern, the result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable from outer function, got {type(result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                print("Executing operator with inner args/kwargs...")
                inner_result = result(*inner_args, **inner_kwargs)
                
                passed, msg = compare_lenstronomy_objects(inner_expected, inner_result)
                if not passed:
                    print(f"TEST FAILED")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function execution
        print("Scenario A detected: Simple function execution")
        
        # First validate the structure
        passed, msg = validate_output_structure(result)
        if not passed:
            print(f"TEST FAILED - Structure validation")
            print(f"Failure message: {msg}")
            sys.exit(1)
        
        print(f"Expected output type: {type(expected_output).__name__}")
        print(f"Actual result type: {type(result).__name__}")
        
        # Compare outputs using custom comparison
        passed, msg = compare_lenstronomy_objects(expected_output, result)
        if not passed:
            print(f"TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()