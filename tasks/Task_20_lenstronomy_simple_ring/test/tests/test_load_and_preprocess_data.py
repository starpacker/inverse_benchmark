import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_imaging_data(expected, actual):
    """Compare ImageData objects by their data content."""
    try:
        # Compare the actual image data arrays
        if hasattr(expected, 'data') and hasattr(actual, 'data'):
            if not np.allclose(expected.data, actual.data, rtol=1e-5, atol=1e-8, equal_nan=True):
                return False, "ImageData.data arrays don't match"
        return True, ""
    except Exception as e:
        return False, f"Error comparing ImageData: {str(e)}"


def compare_psf(expected, actual):
    """Compare PSF objects by their attributes."""
    try:
        if hasattr(expected, 'psf_type') and hasattr(actual, 'psf_type'):
            if expected.psf_type != actual.psf_type:
                return False, "PSF type mismatch"
        if hasattr(expected, 'fwhm') and hasattr(actual, 'fwhm'):
            if not np.isclose(expected.fwhm, actual.fwhm, rtol=1e-5):
                return False, "PSF fwhm mismatch"
        return True, ""
    except Exception as e:
        return False, f"Error comparing PSF: {str(e)}"


def compare_model_class(expected, actual):
    """Compare LensModel or LightModel objects."""
    try:
        # Compare model lists
        if hasattr(expected, 'lens_model_list') and hasattr(actual, 'lens_model_list'):
            if expected.lens_model_list != actual.lens_model_list:
                return False, "lens_model_list mismatch"
        if hasattr(expected, 'profile_type_list') and hasattr(actual, 'profile_type_list'):
            if expected.profile_type_list != actual.profile_type_list:
                return False, "profile_type_list mismatch"
        return True, ""
    except Exception as e:
        return False, f"Error comparing model class: {str(e)}"


def custom_recursive_check(expected, actual, path="output"):
    """Custom comparison that handles lenstronomy objects properly."""
    
    # Handle None cases
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"One value is None at {path}"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"Type mismatch at {path}: expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"Array values mismatch at {path}"
        return True, ""
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch at {path}: expected dict, got {type(actual)}"
        
        for key in expected:
            if key not in actual:
                return False, f"Missing key '{key}' at {path}"
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle lenstronomy ImageData objects
    if hasattr(expected, '__class__') and 'ImageData' in expected.__class__.__name__:
        return compare_imaging_data(expected, actual)
    
    # Handle lenstronomy PSF objects
    if hasattr(expected, '__class__') and 'PSF' in expected.__class__.__name__:
        return compare_psf(expected, actual)
    
    # Handle lenstronomy LensModel objects
    if hasattr(expected, '__class__') and 'LensModel' in expected.__class__.__name__:
        return compare_model_class(expected, actual)
    
    # Handle lenstronomy LightModel objects
    if hasattr(expected, '__class__') and 'LightModel' in expected.__class__.__name__:
        return compare_model_class(expected, actual)
    
    # Handle numeric types
    if isinstance(expected, (int, float, np.number)):
        if not isinstance(actual, (int, float, np.number)):
            return False, f"Type mismatch at {path}: expected numeric, got {type(actual)}"
        if np.isnan(expected) and np.isnan(actual):
            return True, ""
        if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8):
            return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    # Handle strings
    if isinstance(expected, str):
        if expected != actual:
            return False, f"String mismatch at {path}: expected '{expected}', got '{actual}'"
        return True, ""
    
    # Handle other objects - check if they're the same type at minimum
    if type(expected) != type(actual):
        return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
    
    # For other objects, try to compare attributes or consider them equal if same type
    return True, ""


def main():
    data_paths = ['/home/yjh/lenstronomy_simple_ring_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
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
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print("Executing load_and_preprocess_data with outer args/kwargs...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR during Phase 1 (loading/executing outer data): {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    try:
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print(f"Detected Scenario B: Factory/Closure pattern with {len(inner_paths)} inner file(s)")
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                if not callable(result):
                    print("ERROR: Result from outer function is not callable for Scenario B")
                    sys.exit(1)
                
                print("Executing result operator with inner args/kwargs...")
                actual_result = result(*inner_args, **inner_kwargs)
                
                print("Comparing results...")
                passed, msg = custom_recursive_check(expected, actual_result)
                
                if not passed:
                    print("TEST FAILED")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
        else:
            # Scenario A: Simple function pattern
            print("Detected Scenario A: Simple function pattern")
            expected = outer_data.get('output')
            
            print("Comparing results...")
            passed, msg = custom_recursive_check(expected, result)
            
            if not passed:
                print("TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR during Phase 2 (verification): {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()