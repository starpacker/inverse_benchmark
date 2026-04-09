import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_uv_coverage import generate_uv_coverage
from verification_utils import recursive_check

def main():
    """Main test function for generate_uv_coverage"""
    
    data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_generate_uv_coverage.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_uv_coverage.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data
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
    expected_output = outer_data.get('output')
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Handle the rng parameter - we need to recreate it with the same seed
    # The function uses np.random.default_rng(42) when rng=None
    # The captured data likely had rng passed, but we should use None or recreate with seed 42
    
    # Check if rng is in kwargs and handle it
    if 'rng' in outer_kwargs:
        # The original rng state cannot be restored from the serialized object
        # We need to use rng=None to let the function create its own with seed 42
        # OR recreate it with the same seed
        # Based on the gen_data_code, it fixes seeds with 42 before running
        outer_kwargs_clean = {k: v for k, v in outer_kwargs.items() if k != 'rng'}
        outer_kwargs_clean['rng'] = np.random.default_rng(42)
        print("Recreated rng with seed 42")
    else:
        outer_kwargs_clean = outer_kwargs
    
    # Phase 2: Execute the function
    try:
        result = generate_uv_coverage(*outer_args, **outer_kwargs_clean)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Check if this is a factory pattern or simple function
    if inner_paths:
        # Factory/Closure pattern - the result should be callable
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
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
            inner_expected = inner_data.get('output')
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print("Inner function executed successfully")
            except Exception as e:
                print(f"ERROR: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            passed, msg = recursive_check(inner_expected, inner_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print(f"Inner test passed for: {inner_path}")
    else:
        # Simple function pattern
        print("Detected simple function pattern")
        
        # Verify result against expected output
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()