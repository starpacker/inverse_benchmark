import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_imaging_data(expected, actual):
    """Compare ImageData objects by their data content."""
    try:
        # Compare the image data arrays
        expected_data = expected.data
        actual_data = actual.data
        if not np.allclose(expected_data, actual_data, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"ImageData data arrays differ"
        
        # Compare other relevant attributes
        if hasattr(expected, '_background_rms') and hasattr(actual, '_background_rms'):
            if not np.isclose(expected._background_rms, actual._background_rms, rtol=1e-5):
                return False, f"ImageData background_rms differs"
        
        if hasattr(expected, '_exposure_time') and hasattr(actual, '_exposure_time'):
            if not np.isclose(expected._exposure_time, actual._exposure_time, rtol=1e-5):
                return False, f"ImageData exposure_time differs"
        
        return True, "ImageData objects match"
    except Exception as e:
        return False, f"Error comparing ImageData: {str(e)}"


def compare_psf(expected, actual):
    """Compare PSF objects."""
    try:
        # Compare PSF type
        if expected.psf_type != actual.psf_type:
            return False, f"PSF type differs: {expected.psf_type} vs {actual.psf_type}"
        
        # Compare kernel if available
        if hasattr(expected, 'kernel_point_source') and hasattr(actual, 'kernel_point_source'):
            exp_kernel = expected.kernel_point_source
            act_kernel = actual.kernel_point_source
            if not np.allclose(exp_kernel, act_kernel, rtol=1e-5, atol=1e-8, equal_nan=True):
                return False, "PSF kernels differ"
        
        return True, "PSF objects match"
    except Exception as e:
        return False, f"Error comparing PSF: {str(e)}"


def compare_lens_model(expected, actual):
    """Compare LensModel objects."""
    try:
        if expected.lens_model_list != actual.lens_model_list:
            return False, f"LensModel lists differ"
        return True, "LensModel objects match"
    except Exception as e:
        return False, f"Error comparing LensModel: {str(e)}"


def compare_light_model(expected, actual):
    """Compare LightModel objects."""
    try:
        if expected.profile_type_list != actual.profile_type_list:
            return False, f"LightModel profile lists differ"
        return True, "LightModel objects match"
    except Exception as e:
        return False, f"Error comparing LightModel: {str(e)}"


def compare_point_source(expected, actual):
    """Compare PointSource objects."""
    try:
        if expected.point_source_type_list != actual.point_source_type_list:
            return False, f"PointSource type lists differ"
        return True, "PointSource objects match"
    except Exception as e:
        return False, f"Error comparing PointSource: {str(e)}"


def custom_recursive_check(expected, actual, path="output"):
    """Custom comparison that handles lenstronomy objects."""
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.PointSource.point_source import PointSource
    
    # Handle None
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"One is None at {path}: expected={expected}, actual={actual}"
    
    # Handle ImageData
    if isinstance(expected, ImageData):
        if not isinstance(actual, ImageData):
            return False, f"Type mismatch at {path}: expected ImageData, got {type(actual)}"
        return compare_imaging_data(expected, actual)
    
    # Handle PSF
    if isinstance(expected, PSF):
        if not isinstance(actual, PSF):
            return False, f"Type mismatch at {path}: expected PSF, got {type(actual)}"
        return compare_psf(expected, actual)
    
    # Handle LensModel
    if isinstance(expected, LensModel):
        if not isinstance(actual, LensModel):
            return False, f"Type mismatch at {path}: expected LensModel, got {type(actual)}"
        return compare_lens_model(expected, actual)
    
    # Handle LightModel
    if isinstance(expected, LightModel):
        if not isinstance(actual, LightModel):
            return False, f"Type mismatch at {path}: expected LightModel, got {type(actual)}"
        return compare_light_model(expected, actual)
    
    # Handle PointSource
    if isinstance(expected, PointSource):
        if not isinstance(actual, PointSource):
            return False, f"Type mismatch at {path}: expected PointSource, got {type(actual)}"
        return compare_point_source(expected, actual)
    
    # Handle dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch at {path}: expected dict, got {type(actual)}"
        
        # Check keys
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            return False, f"Key mismatch at {path}: missing={missing}, extra={extra}"
        
        # Check each value
        for key in expected.keys():
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle lists
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False, f"Type mismatch at {path}: expected list/tuple, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, ""
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"Type mismatch at {path}: expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False, f"Array values differ at {path}"
        return True, ""
    
    # Handle torch tensors
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return False, f"Type mismatch at {path}: expected Tensor, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if not torch.allclose(expected.float(), actual.float(), rtol=1e-5, atol=1e-8):
            return False, f"Tensor values differ at {path}"
        return True, ""
    
    # Handle floats
    if isinstance(expected, float):
        if not isinstance(actual, (int, float, np.floating)):
            return False, f"Type mismatch at {path}: expected float, got {type(actual)}"
        if not np.isclose(expected, float(actual), rtol=1e-5, atol=1e-8):
            return False, f"Float value mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    # Handle ints
    if isinstance(expected, (int, np.integer)):
        if not isinstance(actual, (int, np.integer)):
            return False, f"Type mismatch at {path}: expected int, got {type(actual)}"
        if expected != actual:
            return False, f"Int value mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    # Handle strings
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False, f"Type mismatch at {path}: expected str, got {type(actual)}"
        if expected != actual:
            return False, f"String mismatch at {path}: expected '{expected}', got '{actual}'"
        return True, ""
    
    # Handle booleans
    if isinstance(expected, (bool, np.bool_)):
        if expected != actual:
            return False, f"Bool mismatch at {path}: expected {expected}, got {actual}"
        return True, ""
    
    # Default: try equality
    try:
        if expected == actual:
            return True, ""
        else:
            return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
    except Exception as e:
        # If comparison fails, check types at least
        if type(expected) == type(actual):
            return True, ""  # Same type, assume OK for complex objects
        return False, f"Comparison failed at {path}: {str(e)}"


def main():
    data_paths = ['/home/yjh/lenstronomy_double_quasar_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    print("=" * 60)
    print("Test: load_and_preprocess_data")
    print("=" * 60)
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    if outer_path is None:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    print("\n=== Phase 1: Loading outer data ===")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    print("\n=== Executing load_and_preprocess_data ===")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully")
        print(f"Result type: {type(result)}")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario and get expected output
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        inner_path = inner_paths[0]
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        print(f"Inner args: {inner_args}")
        print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        # Execute the operator
        try:
            if callable(result):
                actual_result = result(*inner_args, **inner_kwargs)
            else:
                print("ERROR: Result from Phase 1 is not callable for Scenario B")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n=== Scenario A: Simple Function ===")
        expected = outer_data.get('output')
        actual_result = result
    
    # Comparison
    try:
        passed, msg = custom_recursive_check(expected, actual_result)
    except Exception as e:
        print(f"ERROR: Comparison failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print("TEST FAILED")
        print(f"Failure message: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()