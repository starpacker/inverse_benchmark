import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_sphere_intensity import sphere_intensity
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/sasview_saxs_sandbox_sandbox/run_code/std_data/standard_data_sphere_intensity.pkl']
    
    # Classify paths into outer and inner
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_sphere_intensity.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_sphere_intensity.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    # Execute the function
    try:
        agent_result = sphere_intensity(*outer_args, **outer_kwargs)
        print("Successfully called sphere_intensity with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute sphere_intensity: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from sphere_intensity, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Load inner data
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        # Execute the operator with inner data
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        result = agent_result
        expected = outer_output
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        if isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
            print(f"Expected shape: {expected.shape}, Result shape: {result.shape}")
            print(f"Expected dtype: {expected.dtype}, Result dtype: {result.dtype}")
            if expected.shape == result.shape:
                diff = np.abs(expected - result)
                print(f"Max difference: {np.max(diff)}")
                print(f"Mean difference: {np.mean(diff)}")
        sys.exit(1)

if __name__ == "__main__":
    main()