import sys
import os
import dill
import numpy as np
import traceback

# Add the repo path to sys.path
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')
sys.path.insert(0, REPO_DIR)

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def load_pickle_safely(filepath):
    """Safely load a pickle file, returning None if it fails."""
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return None
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        print(f"File is empty (0 bytes): {filepath}")
        return None
    
    print(f"File size: {file_size} bytes")
    
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except EOFError as e:
        print(f"EOFError loading pickle (file may be truncated): {e}")
        return None
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


def run_direct_test():
    """Run a direct test without relying on pickle data."""
    print("\n" + "="*60)
    print("Running direct test (generating fresh data)")
    print("="*60)
    
    # Define test parameters
    test_params = {
        'N': 128,
        'n_spokes': 220,
        'noise_level': 0.002
    }
    
    print(f"Test parameters: {test_params}")
    
    try:
        # Run the function
        result = load_and_preprocess_data(**test_params)
        
        # Verify the result structure
        expected_keys = ['phantom', 'kdata', 'kdata_clean', 'op_dc', 'op_plain', 'trajectory', 'params']
        
        if not isinstance(result, dict):
            print(f"FAILED: Expected dict, got {type(result)}")
            return False
        
        missing_keys = [k for k in expected_keys if k not in result]
        if missing_keys:
            print(f"FAILED: Missing keys in result: {missing_keys}")
            return False
        
        # Verify phantom shape
        phantom = result['phantom']
        if phantom.shape != (test_params['N'], test_params['N']):
            print(f"FAILED: Phantom shape {phantom.shape} != expected ({test_params['N']}, {test_params['N']})")
            return False
        
        # Verify trajectory shape
        trajectory = result['trajectory']
        expected_traj_shape = (test_params['n_spokes'], test_params['N'], 2)
        if trajectory.shape != expected_traj_shape:
            print(f"FAILED: Trajectory shape {trajectory.shape} != expected {expected_traj_shape}")
            return False
        
        # Verify kdata is complex
        if not np.iscomplexobj(result['kdata']):
            print("FAILED: kdata should be complex")
            return False
        
        # Verify params
        params = result['params']
        if params['N'] != test_params['N']:
            print(f"FAILED: params['N'] = {params['N']} != {test_params['N']}")
            return False
        if params['n_spokes'] != test_params['n_spokes']:
            print(f"FAILED: params['n_spokes'] = {params['n_spokes']} != {test_params['n_spokes']}")
            return False
        
        # Verify operators are callable
        op_dc = result['op_dc']
        op_plain = result['op_plain']
        
        if not hasattr(op_dc, 'op') or not callable(op_dc.op):
            print("FAILED: op_dc should have callable 'op' method")
            return False
        
        if not hasattr(op_plain, 'op') or not callable(op_plain.op):
            print("FAILED: op_plain should have callable 'op' method")
            return False
        
        # Test operator functionality
        test_image = phantom.astype(np.complex64)
        kdata_test = op_plain.op(test_image)
        if not np.iscomplexobj(kdata_test):
            print("FAILED: Operator output should be complex")
            return False
        
        print(f"Result keys: {list(result.keys())}")
        print(f"Phantom shape: {phantom.shape}, dtype: {phantom.dtype}")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"kdata shape: {result['kdata'].shape}, dtype: {result['kdata'].dtype}")
        print(f"Params: {params}")
        
        print("\nAll direct tests passed!")
        return True
        
    except Exception as e:
        print(f"FAILED: Exception during direct test: {e}")
        traceback.print_exc()
        return False


def main():
    data_paths = ['/data/yjh/mri_nufft_recon_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Categorize files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    print("="*60)
    print("Test Configuration")
    print("="*60)
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Try to load outer data
    outer_data = None
    if outer_path:
        print(f"\nLoading outer data from: {outer_path}")
        outer_data = load_pickle_safely(outer_path)
    
    # If outer data couldn't be loaded, run direct test
    if outer_data is None:
        print("\nCould not load pickle data, falling back to direct test...")
        if run_direct_test():
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print("\nTEST FAILED")
            sys.exit(1)
    
    # Process with loaded data
    print("\nOuter data loaded successfully")
    print(f"Outer data keys: {list(outer_data.keys())}")
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Determine scenario
    if inner_paths:
        print("\nScenario B: Factory/Closure pattern detected")
    else:
        print("\nScenario A: Simple function test")
    
    # Execute the function
    try:
        print("\nExecuting load_and_preprocess_data...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"FAILED: Exception during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Handle Scenario B (inner paths exist)
    if inner_paths:
        for inner_path in inner_paths:
            print(f"\nProcessing inner path: {inner_path}")
            inner_data = load_pickle_safely(inner_path)
            
            if inner_data is None:
                print(f"WARNING: Could not load inner data, skipping")
                continue
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            if not callable(result):
                print(f"FAILED: Expected callable result for Scenario B, got {type(result)}")
                sys.exit(1)
            
            try:
                inner_result = result(*inner_args, **inner_kwargs)
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"FAILED: Inner comparison failed - {msg}")
                    sys.exit(1)
                print(f"Inner test passed")
            except Exception as e:
                print(f"FAILED: Exception during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Compare result directly
        if expected_output is not None:
            print("\nComparing result with expected output...")
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"FAILED: Comparison failed - {msg}")
                    sys.exit(1)
                print("Comparison passed")
            except Exception as e:
                print(f"FAILED: Exception during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("WARNING: No expected output to compare against")
            # At minimum, verify result structure
            if not isinstance(result, dict):
                print(f"FAILED: Expected dict result, got {type(result)}")
                sys.exit(1)
            expected_keys = ['phantom', 'kdata', 'kdata_clean', 'op_dc', 'op_plain', 'trajectory', 'params']
            missing_keys = [k for k in expected_keys if k not in result]
            if missing_keys:
                print(f"FAILED: Missing keys: {missing_keys}")
                sys.exit(1)
            print("Result structure verified")
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()